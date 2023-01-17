import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option("display.max_columns", 999)
pd.set_option("display.max_rows", 999)
%matplotlib inline
import warnings
warnings.simplefilter(action = "ignore")
df = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")
df.head()
df.columns
df.shape
#almost 81 columns are there.
#first we need to see the nan values there or not
null_lst = []
null_percen = []
null_count = []
for i in df.columns:
    count = 0
    if df[i].isnull().sum()>=1:
        null_lst.append(i)
        count = df[i].isnull().sum()
        null_count.append(count)
        percentage = (count/len(df))*100
        null_percen.append(percentage)
null_dict = {"null_lst" : null_lst, "null_percentage" : null_percen, "null_count" : null_count}
null_df = pd.DataFrame.from_dict(null_dict)
null_df
fig = plt.figure(figsize = (20,5))
sns.barplot(x = "null_lst", y = "null_percentage", data = null_df)
#Need to see the plot based on null values:
for i in null_lst:
    fig = plt.figure()
    df[i + '_null_feature'] = np.where(df[i].isnull(), 1, 0)
    sns.barplot(x = df[i + '_null_feature'], y = df['SalePrice'], data = df)
    df.drop(columns = [i + "_null_feature"], inplace = True)
    plt.show()
#split the cata(nominal and ordinal based on the description) and num variables
categorical_features = df.select_dtypes(include = "object")
numerical_features = df.select_dtypes(exclude = "object")

ordinal_categorical_features = df[['MSZoning', 'LotShape', 'LandSlope', 'HouseStyle', 'BldgType', 'ExterQual', 'ExterCond','BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'HeatingQC', 'KitchenQual', 'FireplaceQu', 'GarageQual', 'GarageFinish', 'GarageCond', 'PoolQC']].columns
nominal_categorical_features = []
for i in categorical_features:
    if i not in ordinal_categorical_features:
        nominal_categorical_features.append(i)
print(categorical_features.shape)
len(ordinal_categorical_features) + len(nominal_categorical_features)
null_cata = []
null_nom_cata = []
null_ord_cata = []
null_num = []
for i in null_lst:
    if i in categorical_features:
        null_cata.append(i)
        if i in nominal_categorical_features:
            null_nom_cata.append(i)
        elif i in ordinal_categorical_features:
            null_ord_cata.append(i)
    elif i in numerical_features:
        null_num.append(i)
for i in null_ord_cata:
    print(i, null_df[null_df["null_lst"] == i])
df['BsmtCond'].value_counts()
#perform label encoding on the cata's
Qual_dict = {np.nan : 0, "Fa" : 1, "TA" : 2, "Gd" : 3, "Ex" : 4}
cond_dict = {np.nan : 0, "Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, 'Ex' : 5}
Exposure_dict = {np.nan : 0, "No" : 1, "Mn" : 2, "Av" : 3, "Gd" : 4}
Fintype_dict = {np.nan : 0, "Unf" : 1, "LwQ" : 2, "Rec" : 3, "BLQ" : 4, "ALQ" : 5, "GLQ" : 6}
Fire_dict = {np.nan : 0, "Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5}
Garage_dict = {np.nan : 0, "Unf" : 1, "RFn" : 2, "Fin" : 3}
df['BsmtQual'] = df['BsmtQual'].map(Qual_dict)
df['BsmtCond'] = df['BsmtCond'].map(cond_dict)
df['GarageQual'] = df['GarageQual'].map(cond_dict)
df['GarageCond'] = df['GarageCond'].map(cond_dict)
df['PoolQC'] = df['PoolQC'].map(Qual_dict)

df['BsmtExposure'] = df['BsmtExposure'].map(Exposure_dict)

df['BsmtFinType1'] = df['BsmtFinType1'].map(Fintype_dict)

df['BsmtFinType2'] = df['BsmtFinType2'].map(Fintype_dict)

df['FireplaceQu'] = df["FireplaceQu"].map(Fire_dict)

df['GarageFinish'] = df["GarageFinish"].map(Garage_dict)


for i in null_ord_cata:
    print(df[i].isnull().sum())
#now need to work on nomical categories:
for i in null_nom_cata:
    print(i, null_df[null_df["null_lst"] == i])
#replace than all nan's with another category:
for i in null_nom_cata:
    df[i] = df[i].fillna("missed")
for i in null_cata:
    val = df[i].isnull().sum()
    if val > 0 :
        print("Still i am null!!")
    elif val == 0:
        print(i , "I am fine!")
for i in null_num:
    print(i, null_df[null_df["null_lst"] == i])
#fill all na's in num with median except lotfrontage and fill that with random values
def random_val_for_cata(df,var):
    random_sample = df[var].dropna().sample(df[i].isnull().sum(), random_state = 0)
    random_sample.index = df[df[var].isnull()].index
    df.loc[df[var].isnull(), var] = random_sample
for i in null_num:
    if i == "LotFrontage":
        #function is taken for fillin random values.
        random_val_for_cata(df,i)
    else:
        df[i] = df[i].fillna(np.median(df[i]))
for i in df.columns:
    if df[i].isnull().sum()>0:
        print(i, df[i].isnull().sum())
#df['GarageYrBlt'] #year is having null value's 
#replace this with highest value
df['GarageYrBlt'] = df['GarageYrBlt'].fillna(df['GarageYrBlt'].value_counts().index.max())
df['MasVnrArea'] = df['MasVnrArea'].fillna(int(df['MasVnrArea'].mean()))
#Now all na's are removed.
for i in categorical_features:
    print(i, df[i].unique())
for i in ordinal_categorical_features:
    print(i, df[i].unique())
mszone_dict = {"C (all)" : 1, "FV" : 2, "RL" : 3,'RM' : 4, 'RH' : 5}
lotshape_dict = {"IR3" : 1, "IR2" : 2, "IR1" : 3, "Reg" : 4}
landslope_dict = {"Gtl" : 1, "Mod" : 2, "Sev" : 3}
house_dict = {"1Story" : 1, "1.5Fin" : 2, "1.5Unf" : 3, "2Story" : 4, "2.5Fin" : 5, "2.5Unf" : 6, "SFoyer" : 7, "SLvl" : 8}
blgd_dict = {"1Fam" : 1, "2fmCon" : 2, "Duplex" : 3, "TwnhsE": 4, "Twnhs" : 5}
qual_dict = {"Po":1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex": 5}
def label_encoding(df,var, dic):
    df[var] = df[var].map(dic)
label_encoding(df,"MSZoning", mszone_dict)
label_encoding(df,"LotShape", lotshape_dict)
label_encoding(df,"LandSlope", landslope_dict)
label_encoding(df,"HouseStyle", house_dict)

label_encoding(df,"BldgType", blgd_dict)
label_encoding(df,"ExterQual",qual_dict)
label_encoding(df,"ExterCond",qual_dict)
label_encoding(df,"HeatingQC",qual_dict)
label_encoding(df,"KitchenQual",qual_dict)
for i in ordinal_categorical_features:
    print(i, df[i].unique())
less_norm_catas = []
for i in nominal_categorical_features:
    if df[i].nunique()<4:
        less_norm_catas.append(i)
print(less_norm_catas)
#for less cata's, we create dummy's
Street = pd.get_dummies(df['Street'], drop_first = True)
Alley = pd.get_dummies(df['Alley'], drop_first = True)
Street = pd.get_dummies(df['Street'], drop_first = True)
Utilities = pd.get_dummies(df['Utilities'], drop_first = True)
CentralAir = pd.get_dummies(df['CentralAir'], drop_first = True)
PavedDrive = pd.get_dummies(df['PavedDrive'], drop_first = True)
df = pd.concat([df,Street, Alley, Utilities, CentralAir, PavedDrive], axis=1)
df.drop(columns = less_norm_catas , inplace = True)
#now work on cata's having more number of categories
more_norm_cata = []
for i in categorical_features:
    if i not in less_norm_catas:
            more_norm_cata.append(i)
more_norm_cata
#replace the cate with their mean based on the saleprice.
for i in more_norm_cata:
    dic = (df.groupby([i])['SalePrice'].mean()/len(df)).to_dict()
    df[i] = df[i].map(dic)
df
for i in numerical_features:
    colors = "#" + str(np.random.randint(100000,999999))
    plt.hist(df[i], color = colors, label = i)

    plt.legend()
    plt.show()
Y = df[['SalePrice']]
df.drop(columns = ['SalePrice'], inplace = True)
#here i am not going to apply any transformations
#Apply minmaxscalar
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df)
df = pd.DataFrame(scaled_data, columns = df.columns)
df
df.drop("Id", axis = 1, inplace = True)
final_df = pd.concat([df, Y], axis = 1)
