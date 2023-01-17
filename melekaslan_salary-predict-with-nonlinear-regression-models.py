import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
from sklearn import model_selection
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn import neighbors
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
#Veri setini okutup ilk 5 satırını gözlemliyoruz.
data = pd.read_csv("../input/hitters/Hitters.csv")
data.head()
import warnings
warnings.simplefilter(action = "ignore")
#Veri setinini boyutunu inceliyoruz. 322 adet gözlem ve 20 değişken bulunmaktadır.
data.shape
#Veri setine istatiktiksel olarak göz atıyoruz.
data.describe().T
#Kategorik değişken olup olmadığına bakıyoruz. Veri setinde 3 adet kategorik değişken bulunuyor.
data.nunique()
categorical_columns = [col for col in data.columns if data[col].dtype == 'object']
categorical_columns
numeric_columns = [c for c in data.columns if c not in categorical_columns]
numeric_columns
#Kategorik değişkenlerin sınıflarından kaçar adet olduğunu inceliyoruz.
print(data["League"].value_counts())
print(data["NewLeague"].value_counts())
print(data["Division"].value_counts())
data.NewLeague.value_counts().plot.barh();
data.Division.value_counts().plot.barh();
data.NewLeague.value_counts().plot.barh();
# Bağımlı değişken olan salary değişkeninin maksimum ve minimum değerlerini buluyoruz.
print("Salary maksimum değer:", data["Salary"].max())
print("Salary minimum değer:", data["Salary"].min())
# Salary değişkeninin histogram ve yoğunluk grafiğini çizdiriyoruz.
sns.distplot(data.Salary);
# Veri setinin korelasyonunu inceleyerek değişkenler arasında ne tür bir ilişki olduğunu anlayabiliriz. 
# Korelasyon değeri > 0 ise pozitif korelasyon bulunmaktadır. Bir değişken değeri artarken diğer değişkenin de değeri artmaktadır.
# Korelasyon = 0 ise değişkenler arasında ilişki yoktur anlamına gelir.
# Korelasyon < 0 ise negatif korelasyon bulunmaktadır. Bir değişken artarken diğer değişken azalmaktadır. 
# Korelasyonlar incelendiğinde salary bağımlı değişkene pozitif korelasyon olarak etkimekte olan 2 değişken bulunmaktadır. 
# Bu değişkenler CRBI ile CRuns değişkenleridir. Bunlar arttıkça Salary (Maaş) değişkeni de artmaktadır.
data.corr()
# Veri setinin korelasyon matrisi grafiğini oluşturuyoruz.
f, ax = plt.subplots(figsize= [20,15])
sns.heatmap(data.corr(), annot=True, fmt=".2f", ax=ax, cmap = "magma" )
ax.set_title("Correlation Matrix", fontsize=20)
plt.show()
#League değişkenine göre groupby yapılıp hangi ligde ortalama ne kadar maaş alındığını buluyoruz.
data.groupby("League").agg({"Salary": "mean"})
#Oyuncunun oynadığı pozisyona göre ortalama ne kadar maaş aldığını buluyoruz.
data.groupby("Division").agg({"Salary": "mean"})
#Veri setinin korelasyonu incelendiğinde en yüksek pozitif yönlü korelasyona CRBI değişkeni sahipti. Bu değişken arttıkça salary değişkenin de artmasını bekleriz.
#CRBI değişkeni kariyeri boyunca yapmış olduğu vuruş sayısıdır. CRBI değişkenine göre gruplayıp salary değişkeninin ortalamasını inceliyoruz.
data.groupby("CRBI").agg({"Salary": "mean"})
#Veri setinde CRBI değişkeninden sonra en yüksek korelasyona CRuns değişkeni sahip. Aynı işlemi onun için de yapıyoruz.
data.groupby("CRuns").agg({"Salary": "mean"})
#Yeni lig ve oyuncunun ligde oynadığı süresine göre gruplanarak oyuncunun aldıkları maaşların ortalamasını buluyoruz.
data.groupby(["NewLeague","Years"]).agg({"Salary": "mean"})
#Kariyeri boyunca isabet sayısı en yüksek olan gözlem birimlerine göre veri setini azalan şekilde sıralayalım. 
data.sort_values("CHits", ascending = False)
#Liglere göre gruplayarak kariyeri boyunca yapılan isabet sayılarının maksimum değerine erişelim.
data.groupby("League").agg({"CHits": "max"})
#Oyuncuların oynadıkları pozisyonlara göre gruplama yaparak kariyeri boyunca isabet sayısının maksimum değerine erişelim.
data.groupby("Division").agg({"CHits": "max"})
#Lig değişkenine göre gruplayarak beyzbol sopası ile yapılan vuruş sayısının maksimum değerlerine erişelim.
data.groupby("League").agg({"AtBat": "max"})
#Lig değişkenine göre gruplayarak liglerde yapılan hataların ortalama değerlerine erişelim.
data.groupby("League").agg({"Errors": "mean"})
#Lig değişkenine göre gruplayarak liglerde yapılan hataların maksimum değerlerine erişelim.
data.groupby("League").agg({"Errors": "max"})
#Oyuncunun ligde oynadığı süresine göre gruplayıp oyuncunun kariyeri boyunca beyzbol sopası ile yapılan vuruş sayısının maksimum değerlerine erişelim.
data.groupby("Years").agg({"CAtBat": "max"})
#Lig değişkenine göre gruplayıp kariyeri boyunca acaba liglerde beyzbol sopası ile ortalama kaç atış gerçekleşmiştir buna erişelim.
data.groupby("League").agg({"CAtBat": "mean"})
data["OrtCAtBat"] = data["CAtBat"] / data["Years"] #Oyuncunun kariyeri boyunca ortalama kaç kez topa vurduğu
data["OrtCHits"] = data["CHits"] / data["Years"] #Oyuncunun kariyeri boyunca ortalama kaç kez isabetli vuruş yaptığı
data["OrtCHmRun"] = data["CHmRun"] / data["Years"] #Oyuncunun kariyeri boyunca ortalama kaç kez en değerli vuruşu yaptığı
data["OrtCruns"] = data["CRuns"] / data["Years"] #Oyuncunun kariyeri boyunca takımına ortalama kaç tane sayı kazandırdığı
data["OrtCRBI"] = data["CRBI"] / data["Years"] #Oyuncunun kariyeri boyunca ortalama kaç tane oyuncuya koşu yaptırdığı
data["OrtCWalks"] = data["CWalks"] / data["Years"] #Oyuncun kariyeri boyunca karşı oyuncuya ortalama kaç kez hata yaptırdığı
data.head()
#Amacımız maaş tahmini yapmak olduğu için maaş değişkeniyle yüksek korelasyon içerisinde olan değişkenlere bakarak tahminleme yapacağız.
data = data.drop(['AtBat','Hits','HmRun','Runs','RBI','Walks','Assists','Errors','PutOuts'], axis=1)
data.head()
#Kategorik değişkenlerin sayısal değerlere dönüştürülmesi için Label Encoding ve One Hot Encoding yöntemleri kullanılmaktadır.
#One Hot Encoding yaparak kategorik değişkenleri sayısal değerlere dönüştürüp dummy değişken tuzağından korunalım.
data = pd.get_dummies(data, columns =  ["Division"], drop_first = True)
data = pd.get_dummies(data, columns =  ["League"], drop_first = True)
data = pd.get_dummies(data, columns =  ["NewLeague"], drop_first = True)
data.head()
#Veri setinde kaç adet eksik değer var?
data.isnull().sum()
#Veri setinde kaç adet dolu değer var?
data.notnull().sum()
import missingno as msno
msno.bar(data);
msno.matrix(data);
#Eksik gözlem olan değerlerin kaç yıllık kariyere sahip olduklarını ve hangi ligde oynadıklarını inceleyelim.
data_eksik = data[data["Salary"].isnull()].head()
data_eksik
#Eksik değerleri veri setinden çıkarıyoruz.
data=data.dropna()
data.shape
sns.boxplot(x = data["Years"]);
Q1 = data["Years"].quantile(0.25)
Q3 = data["Years"].quantile(0.75)
IQR = Q3-Q1
lower = Q1 - 1.5*IQR
upper = Q3 + 1.5*IQR
outliers = (data["Years"] > upper) | (data["Years"] < lower)
data["Years"] = data["Years"][~(outliers)]
sns.boxplot(x = data["Years"]);
sns.boxplot(x = data["CAtBat"]);
Q1 = data["CAtBat"].quantile(0.30)
Q3 = data["CAtBat"].quantile(0.70)
IQR = Q3-Q1
lower = Q1 - 1.5*IQR
upper = Q3 + 1.5*IQR
outliers = (data["CAtBat"] > upper) | (data["CAtBat"] < lower)
data["CAtBat"] = data["CAtBat"][~(outliers)]
sns.boxplot(x = data["CAtBat"]);
sns.boxplot(x = data["CHits"]);
Q1 = data["CHits"].quantile(0.30)
Q3 = data["CHits"].quantile(0.70)
IQR = Q3-Q1
lower = Q1 - 1.5*IQR
upper = Q3 + 1.5*IQR
outliers = (data["CHits"] > upper) | (data["CHits"] < lower)
data["CHits"] = data["CHits"][~(outliers)]
sns.boxplot(x = data["CHits"]);
sns.boxplot(x = data["CHmRun"]);
Q1 = data["CHmRun"].quantile(0.35)
Q3 = data["CHmRun"].quantile(0.65)
IQR = Q3-Q1
lower = Q1 - 1.5*IQR
upper = Q3 + 1.5*IQR
outliers = (data["CHmRun"] > upper) | (data["CHmRun"] < lower)
data["CHmRun"] = data["CHmRun"][~(outliers)]
sns.boxplot(x = data["CHmRun"]);
sns.boxplot(x = data["CRuns"]);
Q1 = data["CRuns"].quantile(0.35)
Q3 = data["CRuns"].quantile(0.65)
IQR = Q3-Q1
lower = Q1 - 1.5*IQR
upper = Q3 + 1.5*IQR
outliers = (data["CRuns"] > upper) | (data["CRuns"] < lower)
data["CRuns"] = data["CRuns"][~(outliers)]
sns.boxplot(x = data["CRuns"]);
sns.boxplot(x = data["CRBI"]);
Q1 = data["CRBI"].quantile(0.30)
Q3 = data["CRBI"].quantile(0.70)
IQR = Q3-Q1
lower = Q1 - 1.5*IQR
upper = Q3 + 1.5*IQR
outliers = (data["CRBI"] > upper) | (data["CRBI"] < lower)
data["CRBI"] = data["CRBI"][~(outliers)]
sns.boxplot(x = data["CRBI"]);
sns.boxplot(x = data["CWalks"]);
Q1 = data["CWalks"].quantile(0.35)
Q3 = data["CWalks"].quantile(0.65)
IQR = Q3-Q1
lower = Q1 - 1.5*IQR
upper = Q3 + 1.5*IQR
outliers = (data["CWalks"] > upper) | (data["CWalks"] < lower)
data["CWalks"] = data["CWalks"][~(outliers)]
sns.boxplot(x = data["CWalks"]);
sns.boxplot(x = data["Salary"]);
Q1 = data["Salary"].quantile(0.25)
Q3 = data["Salary"].quantile(0.75)
IQR = Q3-Q1
lower = Q1 - 1.5*IQR
upper = Q3 + 1.5*IQR
outliers = (data["Salary"] > upper) | (data["Salary"] < lower)
data["Salary"] = data["Salary"][~(outliers)]
sns.boxplot(x = data["Salary"]);
data=data.dropna()
data.shape
normalizer = preprocessing.Normalizer()
y = data["Salary"]
X = data.drop('Salary', axis=1)
cols = X.columns
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(X)
X = pd.DataFrame(X, columns = cols)
X.head()
models = []
models.append(('KNN', KNeighborsRegressor()))
models.append(('SVR', SVR()))
models.append(('CART', DecisionTreeRegressor()))
models.append(('RF', RandomForestRegressor()))
models.append(('GBM', GradientBoostingRegressor()))
models.append(("XGBoost", XGBRegressor()))
models.append(("LightGBM", LGBMRegressor()))
models.append(("CatBoost", CatBoostRegressor(verbose = False)))
y = data["Salary"]
X = data.drop("Salary", axis=1)
cols = X.columns
cols
#Feature Selection
#Wrapper Method
#Backward Elimination
#https://towardsdatascience.com/feature-selection-with-pandas-e3690ad8504b
import statsmodels.api as sm
cols = list(X.columns)
pmax = 1
while (len(cols)>0):
    p= []
    X_1 = X[cols]
    X_1 = sm.add_constant(X_1)
    model = sm.OLS(y,X_1).fit()
    p = pd.Series(model.pvalues.values[1:],index = cols)      
    pmax = max(p)
    feature_with_p_max = p.idxmax()
    if(pmax>0.05):
        cols.remove(feature_with_p_max)
    else:
        break
selected_features_BE = cols
print(selected_features_BE)
y = data["Salary"]
X = data[selected_features_BE]
scaler = StandardScaler()
X = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, 
                                                    y, 
                                                    test_size=0.20, 
                                                    random_state=46)
for name, model in models:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        msg = "%s: (%f)" % (name, rmse)
        print(msg)
knn_params = {"n_neighbors": np.arange(1,30,1)}

knn_model = KNeighborsRegressor()

knn_cv_model = GridSearchCV(knn_model, knn_params, cv = 10).fit(X_train, y_train)
knn_cv_model.best_params_
knn_tuned = KNeighborsRegressor(**knn_cv_model.best_params_).fit(X_train, y_train)
y_pred = knn_tuned.predict(X_test)
knn_tuned_score = np.sqrt(mean_squared_error(y_test, y_pred))
knn_tuned_score
#nonlinear
svr_model = SVR() 

svr_params = {"C": [0.01, 0.1,0.3,0.5,0.8,1,5, 10, 50, 100,500,1000,10000]}

svr_cv_model = GridSearchCV(svr_model, svr_params, cv = 10, n_jobs = -1, verbose =  2).fit(X_train, y_train)
svr_cv_model.best_params_
svr_tuned = SVR(**svr_cv_model.best_params_).fit(X_train, y_train)
y_pred = svr_tuned.predict(X_test)
svr_tuned_score = np.sqrt(mean_squared_error(y_test, y_pred))
svr_tuned_score
cart_params = {"max_depth": [2,3,4,5,6,8,10,20,30,50, 100, 500, 1000,5000,10000],
              "min_samples_split": [2,5,10,20,30,50,100,500,1000,5000,10000]}
cart_model = DecisionTreeRegressor()
cart_cv_model = GridSearchCV(cart_model, cart_params, cv = 10).fit(X_train, y_train)
cart_cv_model.best_params_
cart_tuned = DecisionTreeRegressor(**cart_cv_model.best_params_).fit(X_train, y_train)
y_pred = cart_tuned.predict(X_test)
cart_tuned_score = np.sqrt(mean_squared_error(y_test, y_pred))
cart_tuned_score
rf_params = {"max_depth": [5,10,None],
            "max_features": [2,5,10],
            "n_estimators": [100, 500, 1000],
            "min_samples_split": [2,10,30]}
rf_model = RandomForestRegressor(random_state = 42).fit(X_train, y_train)
rf_cv_model = GridSearchCV(rf_model, rf_params, cv = 10, n_jobs = -1, verbose = 2).fit(X_train, y_train)
rf_cv_model.best_params_
rf_tuned = RandomForestRegressor(max_depth=30,
            max_features=3,
            n_estimators=1000,
            min_samples_split=2).fit(X_train, y_train)
y_pred = rf_tuned.predict(X_test)
rf_tuned_score = np.sqrt(mean_squared_error(y_test, y_pred))
rf_tuned_score
Importance = pd.DataFrame({'Importance':rf_tuned.feature_importances_*100}, 
                          index = cols)


Importance.sort_values(by = 'Importance', 
                       axis = 0, 
                       ascending = True).plot(kind = 'barh', 
                                              color = 'b', )

plt.xlabel('Variable Importance')
plt.gca().legend_ = None
xgb_params = {"learning_rate": [0.1,0.01,1],
             "max_depth": [2,5,8],
             "n_estimators": [100,500,1000],
             "colsample_bytree": [0.3,0.6,1]}
xgb = XGBRegressor()
xgb_cv_model  = GridSearchCV(xgb,xgb_params, cv = 10, n_jobs = -1, verbose = 2).fit(X_train, y_train)
xgb_cv_model.best_params_
xgb_tuned = XGBRegressor(**xgb_cv_model.best_params_).fit(X_train, y_train)
y_pred = xgb_tuned.predict(X_test)
xgb_tuned_score = np.sqrt(mean_squared_error(y_test, y_pred))
xgb_tuned_score
Importance = pd.DataFrame({'Importance':xgb_tuned.feature_importances_*100}, 
                          index = cols)


Importance.sort_values(by = 'Importance', 
                       axis = 0, 
                       ascending = True).plot(kind = 'barh', 
                                              color = 'g', )

plt.xlabel('Variable Importance')
plt.gca().legend_ = None
lgb_model = LGBMRegressor()
lgbm_params = {"learning_rate": [0.01, 0.1, 1],
              "n_estimators": [200,1000,10000],
              "max_depth": [2,5,10],
              "colsample_bytree": [1,0.5,0.3]}
lgbm_cv_model = GridSearchCV(lgb_model, 
                             lgbm_params, 
                             cv = 10, 
                             n_jobs = -1, 
                             verbose =2).fit(X_train, y_train)
lgbm_cv_model.best_params_
lgbm_tuned = LGBMRegressor(learning_rate=0.01,
              n_estimators=300,
              max_depth=5,
              colsample_bytree=1).fit(X_train, y_train)
y_pred = lgbm_tuned.predict(X_test)
lgbm_tuned_score = np.sqrt(mean_squared_error(y_test, y_pred))
lgbm_tuned_score
Importance = pd.DataFrame({'Importance':lgbm_tuned.feature_importances_*100}, 
                          index = cols)


Importance.sort_values(by = 'Importance', 
                       axis = 0, 
                       ascending = True).plot(kind = 'barh', 
                                              color = 'y', )

plt.xlabel('Variable Importance')
plt.gca().legend_ = None
catb_model = CatBoostRegressor(verbose = False)
catb_params = {"iterations": [500,1000,10000],
              "learning_rate": [0.01,0.1,1],
              "depth": [2,6,10]}
catb_cv_model = GridSearchCV(catb_model, 
                           catb_params, 
                           cv = 5, 
                           n_jobs = -1, 
                           verbose = 2).fit(X_train, y_train)
catb_cv_model.best_params_
catb_tuned = CatBoostRegressor(iterations=670,
              learning_rate=0.01,
              depth=6,verbose=False).fit(X_train, y_train)
y_pred = catb_tuned.predict(X_test)
catb_tuned_score = np.sqrt(mean_squared_error(y_test, y_pred))
catb_tuned_score
index = ["KNN_tuned","SVR_tuned","CART_tuned","RF_tuned","XGB_tuned","LGBM_tuned","CATB_tuned"]
tuned_score_data = pd.DataFrame({"Tuned Score":[knn_tuned_score,svr_tuned_score,cart_tuned_score,rf_tuned_score,
                                                xgb_tuned_score,lgbm_tuned_score,catb_tuned_score]})
tuned_score_data.index = index
tuned_score_data

