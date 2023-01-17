import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.impute import KNNImputer
from sklearn.tree import DecisionTreeClassifier
import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter("ignore", category=ConvergenceWarning)
pd.pandas.set_option("display.max_columns", None)
pd.set_option("display.float_format", lambda x: '%.4f' % x)
# Import
train = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")
test = pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")

# Bind rows
df = train.append(test).reset_index()

# Num of dtypes
print("Num of Object Variables:", df.select_dtypes(object).shape[1])
print("Num of Integer Variables:", df.select_dtypes("integer").shape[1])
print("Num of Float Variables:", df.select_dtypes("float").shape[1])

# Dimensions
print("df.shape:"+ str(df.shape), "train.shape:"+ str(train.shape),"test.shape:"+ str(test.shape))

# DataFrame
df.head()
# Train datası üzerinde tespit edilen bazı işlemler. 
train = train[train.GrLivArea * train.TotRmsAbvGrd < 45000]
train = train[train.GrLivArea < 4500]
train = train[train.GarageArea * train.GarageCars < 3700]
train = train[(train.FullBath + (train.HalfBath*0.5) + train.BsmtFullBath + (train.BsmtHalfBath*0.5))<5]
train = train.loc[~(train.SalePrice==392500.0)]
train = train.loc[~((train.SalePrice==275000.0) & (train.Neighborhood=='Crawfor'))]
df.head()
class_freq_num = 25  # Kategorik değişkenleri seçmek için frekans sınır değeri

low_q1 = 0.25  # Aykırı değer bulurken kullanılacak alt quantile sınırı

upper_q3 = 0.75  # Aykırı değer bulurken kullanılacak üst quantile sınırı

correlation_limit = 0.60  # Korelasyon incelenirken kullanılacak olan sınır değer
# Korelasyon
plt.figure(figsize=(8,8))
plt.title('Correlation matrix with SalePrice')
selected_columns = ['OverallQual', 'GarageCars', 'GrLivArea', 'YearBuilt', 'FullBath', '1stFlrSF', 'TotalBsmtSF', 'GarageArea']
a = sns.heatmap(train[selected_columns + ['SalePrice']].corr(), annot=True, square=True)
def check_dataframe(dataframe):
    """
    -> Veriye genel bakış sağlar.
    :param dataframe: Genel bakış yapılacak dataframe
    """

    print("Data Frame Raws Lenght : ", dataframe.shape[0],
          "\nData Frame Columns Lenght : ", dataframe.shape[1])

    print("\nData Frame Columns Names : ", list(dataframe.columns))

    print("\nIs data frame has null value? : ", dataframe.isnull().any())

    print("\nHow many missing values are in which columns? :\n", dataframe.isnull().sum())

    cat_names = [col for col in dataframe.columns if dataframe[col].dtype == "O"]
    num_names = [col for col in dataframe.columns if dataframe[col].dtype != "O"]

    print("\nHow many columns are in the object type? : ", len(cat_names), "\n", cat_names)

    print("\nHow many columns are in the numerical type? : ", len(num_names), "\n", num_names)
def get_categorical_and_numeric_columns(dataframe, exit_columns, number_of_unique_classes=class_freq_num):
    """
    -> Kategorik ve sayısal değişkenleri belirler.
    :param dataframe: İşlem yapılacak dataframe
    :param exit_columns: Dikkate alınmayacak değişken ismi
    :param number_of_unique_classes: Değişkenlerin sınıflarının frekans sınırı
    :return: İlk değer olarak kategorik sınıfların adını, ikinci değer olarak sayısal değişkenlerin adını döndürür.
    """

    categorical_columns = [col for col in dataframe.columns
                           if len(dataframe[col].unique()) <= number_of_unique_classes]

    numeric_columns = [col for col in dataframe.columns if len(dataframe[col].unique()) > number_of_unique_classes
                       and dataframe[col].dtype != "O"
                       and col not in exit_columns]

    return categorical_columns, numeric_columns
def cat_summary(dataframe, categorical_columns, target, plot=False):
    """
    -> Kategorik değişkenlerin sınıflarının oranını ve targettaki medyanı gösterir.
    :param dataframe: İşlem yapılacak dataframe
    :param categorical_columns: Kategorik değişkenlerin adları
    :param target: Dataframe'de ilgilendiğimiz değişken.
    :param plot: Grafik çizdirmek için argüman : True/False
    """
    for col in categorical_columns:
        print(col, " : ", dataframe[col].nunique(), " unique classes.\n")

        print(col, " : ", dataframe[col].value_counts().sum(), "\n")

        print(pd.DataFrame({"COUNT": dataframe[col].value_counts(),
                            "RATIO ( % )": 100 * dataframe[col].value_counts() / len(dataframe),
                            "TARGET_MEDIAN": dataframe.groupby(col)[target].median()}), end="\n\n\n")

        if plot:
            sns.countplot(x=col, data=dataframe)

            plt.show()
def hist_for_numeric_columns(dataframe, numeric_columns):
    """
    -> Sayısal değişkenlerin histogramını çizdirir.
    :param dataframe: İşlem yapılacak dataframe.
    :param numeric_columns: Sayısal değişkenlerin adları
    """
    col_counter = 0

    data = dataframe.copy()

    for col in numeric_columns:
        data[col].hist(bins=20)

        plt.xlabel(col)

        plt.title(col)

        plt.show()

        col_counter += 1

    print(col_counter, "variables have been plotted!")
def find_correlation(dataframe, numeric_columns, target, corr_limit=correlation_limit):
    """
    -> Sayısal değişkenlerin targetla olan korelasyonunu inceler.
    :param dataframe: İşlem yapılacak dataframe
    :param numeric_columns: Sayısal değişken adları
    :param target: Korelasyon ilişkisinde bakılacak hedef değişken
    :param corr_limit: Korelasyon sınırı. Sınırdan aşağısı düşük, yukarısı yüksek korelasyon
    :return: İlk değer düşük korelasyona sahip değişkenler, ikinci değer yüksek korelasyona sahip değişkenler
    """
    high_correlations = []

    low_correlations = []

    for col in numeric_columns:
        if col == target:
            pass

        else:
            correlation = dataframe[[col, target]].corr().loc[col, target]

            if abs(correlation) > corr_limit:
                high_correlations.append(col + " : " + str(correlation))

            else:
                low_correlations.append(col + " : " + str(correlation))

    return low_correlations, high_correlations

def target_summary_with_categorical_columns(dataframe, categorical_columns, target):
    """
    -> Kategorik değişkenlere göre target analizi yapar.
    :param dataframe: İşlem yapılacak dataframe
    :param categorical_columns: Kategorik değişken adları
    :param target: Analizi yapılacak hedef değişkenin adı
    :return:
    """
    for col in categorical_columns:
        if col != target:
            print(pd.DataFrame({"Target_Median": dataframe.groupby(col)[target].median()}), end="\n\n\n")
def target_summary_with_numeric_columns(dataframe, numeric_columns, exit_columns, target):
    """
    -> Sayısal değişkenlere göre target analizi
    :param dataframe: İşlem yapılacak dataframe
    :param numeric_columns: Sayısal değişken adları
    :param exit_columns: Bakılması istenmeyen değişkenin adı
    :param target: Analizi yapılacak hedef değişkenin adı
    :return:
    """

    for col in numeric_columns:
        if col != target or col != exit_columns:
            print(dataframe.groupby(target).agg({col: np.median}), end="\n\n\n")
def outlier_thresholds(dataframe, variable, low_quantile=low_q1, up_quantile=upper_q3):
    """
    -> Verilen değerin alt ve üst aykırı değerlerini hesaplar ve döndürür.
    :param dataframe: İşlem yapılacak dataframe
    :param variable: Aykırı değeri yakalanacak değişkenin adı
    :param low_quantile: Alt eşik değerin hesaplanması için bakılan quantile değeri
    :param up_quantile: Üst eşik değerin hesaplanması için bakılan quantile değeri
    :return: İlk değer olarak verilen değişkenin alt sınır değerini, ikinci değer olarak üst sınır değerini döndürür
    """
    quantile_one = dataframe[variable].quantile(low_quantile)

    quantile_three = dataframe[variable].quantile(up_quantile)

    interquantile_range = quantile_three - quantile_one

    up_limit = quantile_three + 1.5 * interquantile_range

    low_limit = quantile_one - 1.5 * interquantile_range

    return low_limit, up_limit
def has_outliers(dataframe, numeric_columns, plot=False):
    """
    -> Sayısal değişkenlerde aykırı gözlem var mı?
    -> Varsa isteğe göre box plot çizdirme görevini yapar.
    -> Ayrıca aykırı gözleme sahip değişkenlerin ismini göndürür.
    :param dataframe:  İşlem yapılacak dataframe
    :param numeric_columns: Aykırı değerleri bakılacak sayısal değişken adları
    :param plot: Boxplot grafiğini çizdirmek için bool değer alır. True/False
    :return: Aykırı değerlere sahip değişkenlerin adlarını döner
    """
    variable_names = []

    for col in numeric_columns:
        low_limit, up_limit = outlier_thresholds(dataframe, col)

        if dataframe[(dataframe[col] > up_limit) | (dataframe[col] < low_limit)].any(axis=None):
            number_of_outliers = dataframe[(dataframe[col] > up_limit) | (dataframe[col] < low_limit)].shape[0]

            print(col, " : ", number_of_outliers, " aykırı gözlem.")

            variable_names.append(col)

            if plot:
                sns.boxplot(x=dataframe[col])
                plt.show()

    return variable_names

def remove_outliers(dataframe, numeric_columns):
    """
     Dataframede, verilen sayısal değişkenlerin aykırı gözlemlerini siler ve dataframe döner.
    :param dataframe: İşlem yapılacak dataframe
    :param numeric_columns: Aykırı gözlemleri silinecek sayısal değişken adları
    :return: Aykırı gözlemleri silinmiş dataframe döner
    """

    for variable in numeric_columns:
        low_limit, up_limit = outlier_thresholds(dataframe, variable)

        dataframe_without_outliers = dataframe[~((dataframe[variable] < low_limit) | (dataframe[variable] > up_limit))]

    return dataframe_without_outliers
def replace_with_thresholds(dataframe, numeric_columns):
    """
    Baskılama yöntemi
    Silmemenin en iyi alternatifidir.
    Loc kullanıldığından dataframe içinde işlemi uygular.
    :param dataframe: İşlem yapılacak dataframe
    :param numeric_columns: Aykırı değerleri baskılanacak sayısal değişkenlerin adları
    """
    for variable in numeric_columns:
        low_limit, up_limit = outlier_thresholds(dataframe, variable)

        dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit

        dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit
def missing_values_table(dataframe):
    """
     Eksik değerlere sahip değişkenleri gösterir ve bu değerleri döndürür.
    :param dataframe: İşlem yapılacak dataframe
    :return: Eksik değerlere sahip değişkenlerin adlarını döndürür.
    """
    variables_with_na = [col for col in dataframe.columns
                         if dataframe[col].isnull().sum() > 0]

    n_miss = dataframe[variables_with_na].isnull().sum().sort_values(ascending=False)

    ratio = (dataframe[variables_with_na].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)

    # ratio = (100 * dataframe[variables_with_na].isnull().sum() / dataframe.shape[0]).sort_values(ascending=False)

    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=["n_miss", "ratio"])

    print(missing_df)

    return variables_with_na
def missing_vs_target(dataframe, target, variable_with_na):
    """
    Bu fonksiyon, eksik değerlere sahip değişkenlerin target açısından etkisine bakmamızı sağlar.
    Yeni bir değişken oluşturur : incelenen değer + _NA_FLAG
    Bu yeni değişkene, incelenen değişkende eksik gördüğünde 1, eksik yoksa 0 değerlerini atar.
    Daha sonra bu değişkenlere göre gruplama yapıp, target incelenir.
    :param dataframe: İşlem yapılacak dataframe
    :param target: Analizi yapılacak hedef değişkenin adı
    :param variable_with_na: Eksik değerlere sahip değişkenlerin adı.
    """
    temp_df = dataframe.copy()

    for variable in variable_with_na:
        temp_df[variable + "_NA_FLAG"] = np.where(temp_df[variable].isnull(), 1, 0)

    flags_na = temp_df.loc[:, temp_df.columns.str.contains("_NA_")].columns

    for variable in flags_na:
        print(pd.DataFrame({"TARGET_MEDIAN": temp_df.groupby(variable)[target].median()}),
              end="\n\n\n")

def label_encoder(dataframe, categorical_columns):
    """
    2 sınıflı kategorik değişkeni 0-1 yapma
    :param dataframe: İşlem yapılacak dataframe
    :param categorical_columns: Label encode yapılacak kategorik değişken adları
    :return:
    """
    labelencoder = preprocessing.LabelEncoder()

    for col in categorical_columns:

        if dataframe[col].nunique() == 2:
            dataframe[col] = labelencoder.fit_transform(dataframe[col])

    return dataframe
def one_hot_encoder(dataframe, categorical_columns, nan_as_category=False):
    """
    Drop_first doğrusal modellerde yapılması gerekli
    Ağaç modellerde gerekli değil ama yapılabilir.
    dummy_na eksik değerlerden değişken türettirir.
    :param dataframe: İşlem yapılacak dataframe
    :param categorical_columns: One-Hot Encode uygulanacak kategorik değişken adları
    :param nan_as_category: NaN değişken oluştursun mu? True/False
    :return: One-Hot Encode yapılmış dataframe ve bu işlem sonrası oluşan yeni değişken adlarını döndürür.
    """
    original_columns = list(dataframe.columns)

    dataframe = pd.get_dummies(dataframe, columns=categorical_columns,
                               dummy_na=nan_as_category, drop_first=True)

    new_columns = [col for col in dataframe.columns if col not in original_columns]

    return dataframe, new_columns
def rare_analyser(dataframe, categorical_columns, target, rare_perc):
    """
     Data frame değişkenlerinin herhangi bir sınıfı, verilen eşik değerden düşük frekansa sahipse bu değişkenleri gösterir.
    :param dataframe: İşlem yapılacak dataframe
    :param categorical_columns: Rare analizi yapılacak kategorik değişken adları
    :param target: Analizi yapılacak hedef değişken adı
    :param rare_perc: Rare için sınır değer. Altında olanlar rare kategorisine girer.
    :return:
    """
    rare_columns = [col for col in categorical_columns
                    if (dataframe[col].value_counts() / len(dataframe) < rare_perc).any(axis=None)]

    for var in rare_columns:
        print(var, " : ", len(dataframe[var].value_counts()))

        print(pd.DataFrame({"COUNT": dataframe[var].value_counts(),
                            "RATIO": dataframe[var].value_counts() / len(dataframe),
                            "TARGET_MEAN": dataframe.groupby(var)[target].mean(),
                            "TARGET_MEDIAN": dataframe.groupby(var)[target].median()}),
              end="\n\n\n")

    print(len(rare_columns), " adet rare sınıfa sahip değişken var.")

def rare_encoder(dataframe, categorical_columns, rare_perc):
    """
    -> Nadir sınıfları rare olarak dönüştürür.
    -> Verilen kategorik değişkenlerden, rare sınırı altında herhangi bir sınıfı olan değişkenleri yakalar.
    -> Daha sonra bu sınıflar içinde, rare sınırının altında olan sınıfların indexlerini yakalar.
    -> Yakalanan indexleri kullanarak, geçici dataframe içinde bu indexlere sahip olan sınıflara "Rare" yazar.
    -> Oluşan geçici dataframe'i döndürür.
    :param dataframe: İşlem yapılacak dataframe
    :param rare_perc: Rare için sınır değer. Altında olanlar rare kategorisine girer.
    :param categorical_columns: Rare analizi yapılacak kategorik değişken adları
    :return: Rare yazılmış sınıflara sahip geçici dataframe'i döndürür.
    """
    temp_df = dataframe.copy()

    rare_columns = [col for col in categorical_columns
                    if (dataframe[col].value_counts() / len(dataframe) < rare_perc).any(axis=None)]

    for var in rare_columns:
        tmp = temp_df[var].value_counts() / len(temp_df)

        rare_labels = tmp[tmp < rare_perc].index

        temp_df[var] = np.where(temp_df[var].isin(rare_labels), "Rare", temp_df[var])

    return temp_df
def robust_scaler(variable):
    var_median = variable.median()
    quartile1 = variable.quantile(0.25)
    quartile3 = variable.quantile(0.75)
    interquantile_range = quartile3 - quartile1
    if int(interquantile_range) == 0:
        quartile1 = variable.quantile(0.05)
        quartile3 = variable.quantile(0.95)
        interquantile_range = quartile3 - quartile1
        if int(interquantile_range) == 0:
            quartile1 = variable.quantile(0.10)
            quartile3 = variable.quantile(0.99)
            interquantile_range = quartile3 - quartile1
            z = (variable - var_median) / interquantile_range
            return round(z, 3)

        z = (variable - var_median) / interquantile_range
        return round(z, 3)
    else:
        z = (variable - var_median) / interquantile_range
    return round(z, 3)
# Adım 1 - Genel Bakış
check_dataframe(df)
# Adım 2 - Kategorik ve Sayısal Değişkenleri Elde Etme
categorical_columns, numerical_columns = get_categorical_and_numeric_columns(df, "Id")
print("\nCategorical columns : ", categorical_columns,
      "\n\nNumeric Columns : ", numerical_columns)
# Adım 3 - Kategorik Değişken vs Target Analizi
cat_summary(df, categorical_columns, "SalePrice")
# Adım 4 - Sayısal Değişkenler İçin Histogram Çizdirimi
hist_for_numeric_columns(df, numerical_columns)
# Adım 5 - Korelasyon Analizi
low_corr_list, up_corr_list = find_correlation(df, numerical_columns, "SalePrice")
print("\nHighly correlated list : ", len(up_corr_list))
for up in up_corr_list:
    print(up)
print("\nLow correlated list : ", len(low_corr_list))
for low in low_corr_list:
    print(low)
# Adım 6 - Eksik Değer Analizi
miss_values = missing_values_table(df)
# Eksik değerlerin doldurulması
df['Exterior1st'] = df['Exterior1st'].fillna(df['Exterior1st'].mode()[0])

df['MSZoning'] = df.groupby('MSSubClass')['MSZoning'].transform(lambda x: x.fillna(x.mode()[0]))

df["LotFrontage"] = df.groupby("Neighborhood")["LotFrontage"].transform(lambda x: x.fillna(x.median()))

df["Functional"] = df["Functional"].fillna("Typ")

fill_none_col = ["PoolQC","MiscFeature","Alley","Fence","FireplaceQu","GarageType","GarageFinish","GarageQual","GarageCond",
                 "BsmtQual","GarageCond","BsmtQual","BsmtCond","BsmtExposure","BsmtFinType1","BsmtFinType2","MasVnrType","MSSubClass"]
for col in fill_none_col:
    df[col] = df[col].fillna("None")

fill_0_col = ["GarageYrBlt","GarageArea","GarageCars","BsmtFinSF1","BsmtFinSF2","BsmtUnfSF","TotalBsmtSF","BsmtFullBath","BsmtHalfBath","MasVnrArea"]
for col in fill_0_col:
    df[col] = df[col].fillna(0)

fill_mod_col = ["Electrical","KitchenQual","Exterior2nd","SaleType"]
for col in fill_mod_col:
    df[col] = df[col].fillna(df[col].mode()[0])
    
# Eksik değer check
new_miss_values = missing_values_table(df)

# Adım 10 - Rare Analizi
rare_analyser(df, categorical_columns, "SalePrice", 0.02)
df['haspool'] = df['PoolArea'].apply(lambda x: 1 if x > 0 else 0)
df['has2ndfloor'] = df['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)
df['hasgarage'] = df['GarageArea'].apply(lambda x: 1 if x > 0 else 0)
df['hasbsmt'] = df['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)
df['hasfireplace'] = df['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)

# Bu değişkenlerde %96 ve üstünde sadece bir sınıf var. Bu yüzden bunlar bizim için bilgi taşımıyor.
first_drop_list = ["Street",'LotFrontage','GarageYrBlt', "Utilities", "Condition2", "RoofMatl", "PoolQC", "MiscFeature","KitchenAbvGr"]
for col in first_drop_list:
    df.drop(col, axis=1, inplace=True)


new_categorical_columns, new_numerical_columns = get_categorical_and_numeric_columns(df, "Id")
print("\nCategorical columns : ", new_categorical_columns,
      "\n\nNumeric Columns : ", new_numerical_columns)
"""rare_labels = ["TA", "Fa","Gd"]
df["BsmtQual"] = np.where(df["BsmtQual"].isin(rare_labels),"TF",df["BsmtQual"])
rare_labels = ["Gd", "TA"]
df["BsmtCond"] = np.where(df["BsmtCond"].isin(rare_labels),"GT",df["BsmtCond"])
df.BsmtExposure.replace(['Av','Gd','Mn','No'], [1,1,1, 0], inplace=True)
rare_labels = ["ALQ", "BLQ","LwQ","Rec","Unf"]
df["BsmtFinType1"] = np.where(df["BsmtFinType1"].isin(rare_labels),"Rare",df["BsmtFinType1"])
rare_labels = ["LwQ","Rec"]
df["BsmtFinType2"] = np.where(df["BsmtFinType2"].isin(rare_labels),"Rare",df["BsmtFinType2"])
rare_labels = ["ALQ", "GLQ","Unf"]
df["BsmtFinType2"] = np.where(df["BsmtFinType2"].isin(rare_labels),"R",df["BsmtFinType2"])
rare_labels = ["Unf", "RFn"]
df["GarageFinish"] = np.where(df["GarageFinish"].isin(rare_labels),"Urf",df["GarageFinish"])
rare_labels = ["Fa", "Po"]
df["GarageQual"] = np.where(df["GarageQual"].isin(rare_labels),"R",df["GarageQual"])
rare_labels = ["Gd", "Ex"]
df["GarageQual"] = np.where(df["GarageQual"].isin(rare_labels),"ge",df["GarageQual"])
rare_labels = ["Attchd", "BuiltIn"]
df["GarageType"] = np.where(df["GarageType"].isin(rare_labels),"AB",df["GarageType"])
rare_labels = ["CarPort", "Detchd"]
df["GarageType"] = np.where(df["GarageType"].isin(rare_labels),"cd",df["GarageType"])
rare_labels = [3,4]
df["GarageCars"] = np.where(df["GarageCars"].isin(rare_labels),3,df["GarageCars"])
rare_labels = [1,2]
df["GarageCars"] = np.where(df["GarageCars"].isin(rare_labels),1,df["GarageCars"])
rare_labels = ["Ex", "TA","Gd"]
df["ExterCond"] = np.where(df["ExterCond"].isin(rare_labels),"EGT",df["ExterCond"])
rare_labels = ["Fa","Po"]
df["ExterCond"] = np.where(df["ExterCond"].isin(rare_labels),"FP",df["ExterCond"])
rare_labels = ["Floor","Wall","Grav"]
df["Heating"] = np.where(df["Heating"].isin(rare_labels),"FWG",df["Heating"])
rare_labels = ["Lvl","Bnk"]
df["LandContour"] = np.where(df["LandContour"].isin(rare_labels),"LB",df["LandContour"])
rare_labels = ["CulDSac","FR3"]
df["LotConfig"] = np.where(df["LotConfig"].isin(rare_labels),"CF",df["LotConfig"])
rare_labels = ["Corner","FR2","Inside"]
df["LotConfig"] = np.where(df["LotConfig"].isin(rare_labels),"CF2I",df["LotConfig"])
# MSZoning - RH, RMY'ye eklenir
df.loc[df["MSZoning"] == "RH", ["MSZoning"]] = "RM"
# LotShape - IR3, IR2 ye katılabilir. IR3 düzensiz, IR2 orta düzensiz demek. IR2-3 IR1 ile birleşsin
df.loc[df["LotShape"] == "IR3", ["LotShape"]] = "IR1"
df.loc[df["LotShape"] == "IR2", ["LotShape"]] = "IR1"
# LandSlope - Sev, Mod a katılabilir. Sev arazi şiddetli eğimli, Mod orta eğimli
df.loc[df["LandSlope"] == "Sev", ["LandSlope"]] = "Mod"
# Neighborhood dikkatli incelemek lazım. Mülkiyetin Ames şehrindeki konumu
# df.loc[df["Neighborhood"] == "Sev"] = "Mod"
# Condition1 - PosA, PosN ile birleşebilir. RRAn ile RRAe birleşebilir. Biri kuzey-güney demir yoluna bitişiklik, diğeri Doğubatı demiriyoluna
df.loc[df["Condition1"] == "PosA", ["Condition1"]] = "NEW_PosAN"
df.loc[df["Condition1"] == "PosN", ["Condition1"]] = "NEW_PosAN"
df.loc[df["Condition1"] == "RRNe", ["Condition1"]] = "NEW_RRANe"
df.loc[df["Condition1"] == "RRAe", ["Condition1"]] = "NEW_RRANe"
df.loc[df["Condition1"] == "RRAn", ["Condition1"]] = "NEW_RRANn"
df.loc[df["Condition1"] == "RRNn", ["Condition1"]] = "NEW_RRANn"
# HouseStyle - 2Story + 2.5 Fin birleşecek. 1.5 lerde bi ekleme yapılabilir
df.loc[df["HouseStyle"] == "2.5Fin", ["HouseStyle"]] = "2Story"
# OverallQual - 9 ve 10 birleşebilir. 1 2 ve 3, 4 ile birleşebilir. Genel malzeme ve bitiş değerlendirmesi. 10 en iyi
df.loc[df["OverallQual"] == 10, ["OverallQual"]] = 8
df.loc[df["OverallQual"] == 9, ["OverallQual"]] = 8
df.loc[df["OverallQual"] == 1, ["OverallQual"]] = 4
df.loc[df["OverallQual"] == 2, ["OverallQual"]] = 4
df.loc[df["OverallQual"] == 3, ["OverallQual"]] = 4
# RoofStyle - Shed silinebilir. Mansard, flat, hipe eklenebilir. Gambdrel Gable ye eklenebilir.
df.loc[df["RoofStyle"] == "Mansard", ["RoofStyle"]] = "Hip"
df.loc[df["RoofStyle"] == "Flat", ["RoofStyle"]] = "Hip"
df.loc[df["RoofStyle"] == "Gambrel", ["RoofStyle"]] = "Gable"
# MasVnrType - BrkCmn, None a eklenir.
df.loc[df["MasVnrType"] == "BrkCmn", ["MasVnrType"]] = "None"
# ExterQual - Fa TA ya eklenir
df.loc[df["ExterQual"] == "Fa", ["ExterQual"]] = "TA"
# Foundation - Stone,Slab, CBlock a eklenir.
df.loc[df["Foundation"] == "Stone", ["Foundation"]] = "CBlock"
df.loc[df["Foundation"] == "Slab", ["Foundation"]] = "CBlock"
# BedroomAbvGr - 0 ve 8 4e eklenir. 5 3 e eklenir. 6 2, 1 e eklenir
df.loc[df["BedroomAbvGr"] == 8, ["BedroomAbvGr"]] = 4
df.loc[df["BedroomAbvGr"] == 5, ["BedroomAbvGr"]] = 4
df.loc[df["BedroomAbvGr"] == 6, ["BedroomAbvGr"]] = 4
df.loc[df["BedroomAbvGr"] == 2, ["BedroomAbvGr"]] = 1
# TotRmsAbvGrd - 2 silinir. 11 10a eklenir. 14 8 e eklenir. 3 4e eklenir
df.loc[df["TotRmsAbvGrd"] == 11, ["TotRmsAbvGrd"]] = 10
df.loc[df["TotRmsAbvGrd"] == 14, ["TotRmsAbvGrd"]] = 8
df.loc[df["TotRmsAbvGrd"] == 3, ["TotRmsAbvGrd"]] = 4
# TotRmsAbvGrd -
df = df[~(df["TotRmsAbvGrd"] == 0)]
df.loc[df["TotRmsAbvGrd"] == 11, ["TotRmsAbvGrd"]] = 10
df.loc[df["TotRmsAbvGrd"] == 14, ["TotRmsAbvGrd"]] = 8
df.loc[df["TotRmsAbvGrd"] == 12, ["TotRmsAbvGrd"]] = 8
# Functional - Typ hariç hepsi birleşecek
df.loc[df["Functional"] == "Maj1", ["Functional"]] = "NEW_Others"
df.loc[df["Functional"] == "Maj2", ["Functional"]] = "NEW_Others"
df.loc[df["Functional"] == "Min1", ["Functional"]] = "NEW_Others"
df.loc[df["Functional"] == "Min2", ["Functional"]] = "NEW_Others"
df.loc[df["Functional"] == "Mod", ["Functional"]] = "NEW_Others"
# FireplaceQu - Po No ya eklenecek. Fa Ta ya eklenecek. Ex Gd ye eklenecek
df.loc[df["FireplaceQu"] == "Po", ["FireplaceQu"]] = "No"
df.loc[df["FireplaceQu"] == "Fa", ["FireplaceQu"]] = "TA"
df.loc[df["FireplaceQu"] == "Ex", ["FireplaceQu"]] = "Gd"
# GarageCond - Po, Fa No ya eklenecek. Ex Gd TA ya eklenecek
df.loc[df["GarageCond"] == "Po", ["GarageCond"]] = "No"
df.loc[df["GarageCond"] == "Fa", ["GarageCond"]] = "No"
df.loc[df["GarageCond"] == "Ex", ["GarageCond"]] = "TA"
df.loc[df["GarageCond"] == "Gd", ["GarageCond"]] = "TA"
# Fence - MnWw MnPrv ye eklenecek
df.loc[df["Fence"] == "GdPrv", ["Fence"]] = "NEW_Fence"
df.loc[df["Fence"] == "GdWo", ["Fence"]] = "NEW_Fence"
df.loc[df["Fence"] == "MnPrv", ["Fence"]] = "NEW_Fence"
df.loc[df["Fence"] == "MnWw", ["Fence"]] = "NEW_Fence"
# SaleType - ConLı, ConLw COD a eklenecek. Con Newe eklenecek. CWD WD ye eklenecek
df.loc[df["SaleType"] == "ConLI", ["SaleType"]] = "Con"
df.loc[df["SaleType"] == "ConLw", ["SaleType"]] = "Con"
df.loc[df["SaleType"] == "ConLD", ["SaleType"]] = "Con"
df.loc[df["SaleType"] == "Oth", ["SaleType"]] = "Con"
df.loc[df["SaleType"] == "CWD", ["SaleType"]] = "WD"
# SaleCondition - Adjland Abnormala eklenecek. Alloca ve Family Normal e eklenecek
df.loc[df["SaleCondition"] == "AdjLand", ["SaleCondition"]] = "Normal"
df.loc[df["SaleCondition"] == "Alloca", ["SaleCondition"]] = "Abnorml"
df.loc[df["SaleCondition"] == "Family", ["SaleCondition"]] = "Normal"
"""

last_miss = missing_values_table(df)
df.head()
df['New_TotalSF'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']
df["NEW_TotalBathRoom"] = df["FullBath"] + df["HalfBath"]*0.5 + df["BsmtFullBath"] + df["BsmtHalfBath"]*0.5
df["NEW_TotalFeet"]= df["GrLivArea"] + df["TotalBsmtSF"]
df["New_Relation"] = (df["OpenPorchSF"]+df["EnclosedPorch"] + df["ScreenPorch"] + df["WoodDeckSF"]) / (df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF'])
df_temp = df

df, new_cols_ohe = one_hot_encoder(df, new_categorical_columns)

outlier_columns = has_outliers(df, new_numerical_columns)

replace_with_thresholds(df, new_numerical_columns)
like_num = [col for col in df.columns if df[col].dtypes != 'O' and len(df[col].value_counts()) < 20]
cols_need_scale = [col for col in df.columns if col not in new_cols_ohe
                   and col not in "Id"
                   and col not in "SalePrice"
                   and col not in like_num]

for col in cols_need_scale:
    df[col] = robust_scaler(df[col])
train_df = df[df['SalePrice'].notnull()]
test_df = df[df['SalePrice'].isnull()]

train_df.to_pickle("train_df.pkl")
test_df.to_pickle("test_df.pkl")
X_train = train_df.drop('SalePrice', axis=1)
y_train = train_df[["SalePrice"]]

X_test = test_df.drop('SalePrice', axis=1)
y_test = test_df[["SalePrice"]]

t = df[df["Id"]>1460]
t= t.drop("SalePrice", axis=1)
len(t)

Ridge_model = Ridge()
Ridge_model.fit(X_train, y_train)
predictions = Ridge_model.predict(X_test)

C = []

for i in predictions:
    C.append(i[0])

my_submission = pd.DataFrame({'Id': t.Id,'SalePrice': C})
my_submission.to_csv('submission.csv', index=False)