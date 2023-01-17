# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
scaler = StandardScaler()
normalizer = preprocessing.Normalizer()

#Veri setini okuma
hitters_data = pd.read_csv("/kaggle/input/hitters/Hitters.csv")
df = hitters_data.copy()
df.head()
#Kısa bilgiler
df.describe().T
#Veri boyutu
df.shape
#Kaç eksik değer var?
df.isnull().sum()
#Eksik verilere bakış
df[df["Salary"].isnull()].head()
#Eksik verileri siliyoruz.
df.dropna(inplace = True)
df.shape
#Veri setinde sadece CHits değişkenini kullanmak için seçiyoruz.
X = df[["CHits"]]
y = df[["Salary"]]
X = scaler.fit_transform(X)
reg_model = LinearRegression()
reg_model.fit(X, y)
print("Formula: "+str(reg_model.intercept_)+" "+str(reg_model.coef_)+"x1")
y_pred = reg_model.predict(X)
from sklearn.metrics import mean_squared_error
sonuc_tum_veri_slr = np.sqrt(mean_squared_error(y, y_pred))
sonuc_tum_veri_slr
sonuc_tum_veri_cv_slr = np.sqrt(np.mean(-cross_val_score(reg_model, X, y, cv = 10, scoring = "neg_mean_squared_error")))
sonuc_tum_veri_cv_slr
X = df[["CHits"]]
y = df[["Salary"]]
X = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.20, 
                                                    random_state=46)
reg_model = LinearRegression()
reg_model.fit(X_train, y_train)
print("Formula: "+str(reg_model.intercept_)+" "+str(reg_model.coef_)+"x1")
y_pred = reg_model.predict(X_train)
sonuc_train_veri_slr = np.sqrt(mean_squared_error(y_train, y_pred))
sonuc_train_veri_slr
y_pred = reg_model.predict(X_test)
sonuc_test_veri_slr = np.sqrt(mean_squared_error(y_test, y_pred))
sonuc_test_veri_slr
sonuc_tt_veri_cv_slr = np.sqrt(np.mean(-cross_val_score(reg_model, X_train, y_train, cv = 10, scoring = "neg_mean_squared_error")))
sonuc_tt_veri_cv_slr
#Kategorik değişkenleri 1-0 şeklinde ifade ederiz
df = pd.get_dummies(df, columns = ['League', 'Division', 'NewLeague'], drop_first = True)
y = df["Salary"]
X = df.drop('Salary', axis=1)
X = scaler.fit_transform(X)
reg_model = LinearRegression()
reg_model.fit(X, y)
print(reg_model.intercept_)
print(reg_model.coef_)
y_pred = reg_model.predict(X)
sonuc_tum_veri_clr = np.sqrt(mean_squared_error(y, y_pred))
sonuc_tum_veri_clr
sonuc_tum_veri_cv_clr = np.sqrt(np.mean(-cross_val_score(reg_model, X, y, cv = 10, scoring = "neg_mean_squared_error")))
sonuc_tum_veri_cv_clr
y = df["Salary"]
X = df.drop('Salary', axis=1)
X = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.20, 
                                                    random_state=46)
reg_model = LinearRegression()
reg_model.fit(X_train, y_train)
print(reg_model.intercept_)
print(reg_model.coef_)
y_pred = reg_model.predict(X_train)
sonuc_train_veri_clr = np.sqrt(mean_squared_error(y_train, y_pred))
sonuc_train_veri_clr
y_pred = reg_model.predict(X_test)
sonuc_test_veri_clr = np.sqrt(mean_squared_error(y_test, y_pred))
sonuc_test_veri_clr
sonuc_tt_veri_cv_clr = np.sqrt(np.mean(-cross_val_score(reg_model, X_train, y_train, cv = 10, scoring = "neg_mean_squared_error")))
sonuc_tt_veri_cv_clr
y = df["Salary"]
X = df.drop('Salary', axis=1)
X = scaler.fit_transform(X)
reg_model = Ridge()
reg_model.fit(X, y)
print(reg_model.intercept_)
print(reg_model.coef_)
y_pred = reg_model.predict(X)
sonuc_tum_veri_ridge = np.sqrt(mean_squared_error(y, y_pred))
sonuc_tum_veri_ridge
sonuc_tum_veri_cv_ridge = np.sqrt(np.mean(-cross_val_score(reg_model, X, y, cv = 10, scoring = "neg_mean_squared_error")))
sonuc_tum_veri_cv_ridge
y = df["Salary"]
X = df.drop('Salary', axis=1)
X = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.20, 
                                                    random_state=46)
reg_model = Ridge()
reg_model.fit(X_train, y_train)
print(reg_model.intercept_)
print(reg_model.coef_)
y_pred = reg_model.predict(X_train)
sonuc_train_veri_ridge = np.sqrt(mean_squared_error(y_train, y_pred))
sonuc_train_veri_ridge
y_pred = reg_model.predict(X_test)
sonuc_test_veri_ridge = np.sqrt(mean_squared_error(y_test, y_pred))
sonuc_test_veri_ridge
sonuc_tt_veri_cv_ridge = np.sqrt(np.mean(-cross_val_score(reg_model, X_train, y_train, cv = 10, scoring = "neg_mean_squared_error")))
sonuc_tt_veri_cv_ridge
y = df["Salary"]
X = df.drop('Salary', axis=1)
X = scaler.fit_transform(X)
reg_model =Lasso()
reg_model.fit(X, y)
print(reg_model.intercept_)
print(reg_model.coef_)
y_pred = reg_model.predict(X)
sonuc_tum_veri_lasso = np.sqrt(mean_squared_error(y, y_pred))
sonuc_tum_veri_lasso
sonuc_tum_veri_cv_lasso = np.sqrt(np.mean(-cross_val_score(reg_model, X, y, cv = 10, scoring = "neg_mean_squared_error")))
sonuc_tum_veri_cv_lasso
y = df["Salary"]
X = df.drop('Salary', axis=1)
X = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.20, 
                                                    random_state=46)
reg_model =Lasso()
reg_model.fit(X_train, y_train)
print(reg_model.intercept_)
print(reg_model.coef_)
y_pred = reg_model.predict(X_train)
sonuc_train_veri_lasso = np.sqrt(mean_squared_error(y_train, y_pred))
sonuc_train_veri_lasso
y_pred = reg_model.predict(X_test)
sonuc_test_veri_lasso = np.sqrt(mean_squared_error(y_test, y_pred))
sonuc_test_veri_lasso
sonuc_tt_veri_cv_lasso = np.sqrt(np.mean(-cross_val_score(reg_model, X_train, y_train, cv = 10, scoring = "neg_mean_squared_error")))
sonuc_tt_veri_cv_lasso
y = df["Salary"]
X = df.drop('Salary', axis=1)
X = scaler.fit_transform(X)
reg_model =ElasticNet()
reg_model.fit(X, y)
print(reg_model.intercept_)
print(reg_model.coef_)
y_pred = reg_model.predict(X)
sonuc_tum_veri_enet = np.sqrt(mean_squared_error(y, y_pred))
sonuc_tum_veri_enet
sonuc_tum_veri_cv_enet = np.sqrt(np.mean(-cross_val_score(reg_model, X, y, cv = 10, scoring = "neg_mean_squared_error")))
sonuc_tum_veri_cv_enet
y = df["Salary"]
X = df.drop('Salary', axis=1)
X = scaler.fit_transform(X)
for i in range(len(X)):
    if X[i][13]<0:
        X[i][13]=0
    else:
        X[i][13]>0
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.20, 
                                                    random_state=46)
reg_model =ElasticNet()
reg_model.fit(X_train, y_train)
print(reg_model.intercept_)
print(reg_model.coef_)
y_pred = reg_model.predict(X_train)
sonuc_train_veri_enet = np.sqrt(mean_squared_error(y_train, y_pred))
sonuc_train_veri_enet
y_pred = reg_model.predict(X_test)
sonuc_test_veri_enet = np.sqrt(mean_squared_error(y_test, y_pred))
sonuc_test_veri_enet
sonuc_tt_veri_cv_enet = np.sqrt(np.mean(-cross_val_score(reg_model, X_train, y_train, cv = 10, scoring = "neg_mean_squared_error")))
sonuc_tt_veri_cv_enet
#İlk sonuçların çıktısı
basicsonuc_df = pd.DataFrame({"All Data Error":[sonuc_tum_veri_slr,sonuc_tum_veri_clr,sonuc_tum_veri_ridge,sonuc_tum_veri_lasso,sonuc_tum_veri_enet],
                              "All Data cvScore(K-Fold)":[sonuc_tum_veri_cv_slr,sonuc_tum_veri_cv_clr,sonuc_tum_veri_cv_ridge,sonuc_tum_veri_cv_lasso,sonuc_tum_veri_cv_enet],
                              "Train Error":[sonuc_train_veri_slr,sonuc_train_veri_clr,sonuc_train_veri_ridge,sonuc_train_veri_lasso,sonuc_train_veri_enet],
                              "Test Error":[sonuc_test_veri_slr,sonuc_test_veri_clr,sonuc_test_veri_ridge,sonuc_test_veri_lasso,sonuc_test_veri_enet],                
                              "Train cvScore(K-Fold)":[sonuc_tt_veri_cv_slr,sonuc_tt_veri_cv_clr,sonuc_tt_veri_cv_ridge,sonuc_tt_veri_cv_lasso,sonuc_tt_veri_cv_enet]})
basicsonuc_df.index= ["SLR", "CLR","RID","LAS","ENE"]
basicsonuc_df
df = hitters_data.copy()
df.head()
#Yeni değişkenler oluşturuyoruz
catbat=df["CAtBat"]/df["Years"]
chits=df["CHits"]/df["Years"]
chmrun=df["CHmRun"]/df["Years"]
cruns=df["CRuns"]/df["Years"]
crbi=df["CRBI"]/df["Years"]
cwalks=df["CWalks"]/df["Years"]
df_seckin = pd.DataFrame({"ortAtBat":catbat,"ortHits":chits,"ortHmRun":chmrun,"ortRuns":cruns,"ortRBI":crbi,"ortWalks":cwalks})
df = pd.concat([df, df_seckin], axis=1)
df.head()
df.corr()
df = df.drop(['AtBat','Hits','HmRun','Runs','RBI','Walks','Assists','Errors',"PutOuts",'League','NewLeague'],axis=1)
df = pd.get_dummies(df, columns =["Division"], drop_first = True)
df.head()
df.isnull().sum()
from sklearn.impute import KNNImputer
imputer = KNNImputer(n_neighbors = 5)
df_filled = imputer.fit_transform(df)
df = pd.DataFrame(df_filled,columns = df.columns)
Q1 = df.Salary.quantile(0.25)
Q3 = df.Salary.quantile(0.75)
IQR = Q3-Q1
lower = Q1 - 1.5*IQR
upper = Q3 + 1.5*IQR
df.loc[df["Salary"] > upper,"Salary"] = upper
from sklearn.neighbors import LocalOutlierFactor
lof =LocalOutlierFactor(n_neighbors= 20)
lof.fit_predict(df)
df_scores = lof.negative_outlier_factor_
np.sort(df_scores)[0:30]
th = np.sort(df_scores)[9]
th
outlier = df_scores > th
df = df[df_scores > th]
df.shape
from sklearn.linear_model import LinearRegression
X = df[["ortHits"]]
y = df[["Salary"]]
X = scaler.fit_transform(X)
reg_model = LinearRegression()
reg_model.fit(X, y)
print("Formula: "+str(reg_model.intercept_)+" "+str(reg_model.coef_)+"x1")
y_pred = reg_model.predict(X)
from sklearn.metrics import mean_squared_error
sonuc_tum_veri_detay_slr = np.sqrt(mean_squared_error(y, y_pred))
sonuc_tum_veri_detay_slr
sonuc_tum_veri_cv_detay_slr = np.sqrt(np.mean(-cross_val_score(reg_model, X, y, cv = 10, scoring = "neg_mean_squared_error")))
sonuc_tum_veri_cv_detay_slr
X = df[["ortHits"]]
y = df[["Salary"]]
X = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.20, 
                                                    random_state=46)
reg_model = LinearRegression()
reg_model.fit(X_train, y_train)
print("Formula: "+str(reg_model.intercept_)+" "+str(reg_model.coef_)+"x1")
y_pred = reg_model.predict(X_train)
sonuc_train_veri_detay_slr = np.sqrt(mean_squared_error(y_train, y_pred))
sonuc_train_veri_detay_slr
y_pred = reg_model.predict(X_test)
sonuc_test_veri_detay_slr = np.sqrt(mean_squared_error(y_test, y_pred))
sonuc_test_veri_detay_slr
sonuc_tt_veri_cv_detay_slr = np.sqrt(np.mean(-cross_val_score(reg_model, X_train, y_train, cv = 10, scoring = "neg_mean_squared_error")))
sonuc_tt_veri_cv_detay_slr
#Geri kalan tüm modellemelerde bu X, y, X_train ve y_train i  kullanacağız.
y = df["Salary"]
X = df.drop("Salary",axis=1)
X = scaler.fit_transform(X)
for i in range(len(X)):
    if X[i][13]<0:
        X[i][13]=0
    else:
        X[i][13]=1  
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.20, 
                                                    random_state=46)
reg_model = LinearRegression()
reg_model.fit(X, y)
reg_model.intercept_
reg_model.coef_
y_pred = reg_model.predict(X)
sonuc_tum_veri_detay_clr = np.sqrt(mean_squared_error(y, y_pred))
sonuc_tum_veri_detay_clr
sonuc_tum_veri_cv_detay_clr = np.sqrt(np.mean(-cross_val_score(reg_model, X, y, cv = 10, scoring = "neg_mean_squared_error")))
sonuc_tum_veri_cv_detay_clr
reg_model = LinearRegression()
reg_model.fit(X_train, y_train)
reg_model.intercept_
reg_model.coef_
y_pred = reg_model.predict(X_train)
sonuc_train_veri_detay_clr = np.sqrt(mean_squared_error(y_train, y_pred))
sonuc_train_veri_detay_clr
y_pred = reg_model.predict(X_test)
sonuc_test_veri_detay_clr = np.sqrt(mean_squared_error(y_test, y_pred))
sonuc_test_veri_detay_clr
sonuc_tt_veri_cv_detay_clr = np.sqrt(np.mean(-cross_val_score(reg_model, X_train, y_train, cv = 10, scoring = "neg_mean_squared_error")))
sonuc_tt_veri_cv_detay_clr
reg_model = Ridge()
reg_model.fit(X, y)
reg_model.intercept_
reg_model.coef_
y_pred = reg_model.predict(X)
sonuc_tum_veri_detay_ridge = np.sqrt(mean_squared_error(y, y_pred))
sonuc_tum_veri_detay_ridge
sonuc_tum_veri_cv_detay_ridge = np.sqrt(np.mean(-cross_val_score(reg_model, X, y, cv = 10, scoring = "neg_mean_squared_error")))
sonuc_tum_veri_cv_detay_ridge
reg_model = Ridge()
reg_model.fit(X_train, y_train)
reg_model.intercept_
reg_model.coef_
y_pred = reg_model.predict(X_train)
sonuc_train_veri_detay_ridge = np.sqrt(mean_squared_error(y_train, y_pred))
sonuc_train_veri_detay_ridge
y_pred = reg_model.predict(X_test)
sonuc_test_veri_detay_ridge = np.sqrt(mean_squared_error(y_test, y_pred))
sonuc_test_veri_detay_ridge
sonuc_tt_veri_cv_detay_ridge = np.sqrt(np.mean(-cross_val_score(reg_model, X_train, y_train, cv = 10, scoring = "neg_mean_squared_error")))
sonuc_tt_veri_cv_detay_ridge
reg_model = Lasso()
reg_model.fit(X, y)
reg_model.intercept_
reg_model.coef_
y_pred = reg_model.predict(X)
sonuc_tum_veri_detay_lasso = np.sqrt(mean_squared_error(y, y_pred))
sonuc_tum_veri_detay_lasso
sonuc_tum_veri_cv_detay_lasso = np.sqrt(np.mean(-cross_val_score(reg_model, X, y, cv = 10, scoring = "neg_mean_squared_error")))
sonuc_tum_veri_cv_detay_lasso
reg_model = Lasso()
reg_model.fit(X_train, y_train)
reg_model.intercept_
reg_model.coef_
y_pred = reg_model.predict(X_train)
sonuc_train_veri_detay_lasso = np.sqrt(mean_squared_error(y_train, y_pred))
sonuc_train_veri_detay_lasso
y_pred = reg_model.predict(X_test)
sonuc_test_veri_detay_lasso = np.sqrt(mean_squared_error(y_test, y_pred))
sonuc_test_veri_detay_lasso
sonuc_tt_veri_cv_detay_lasso = np.sqrt(np.mean(-cross_val_score(reg_model, X_train, y_train, cv = 10, scoring = "neg_mean_squared_error")))
sonuc_tt_veri_cv_detay_lasso
reg_model = ElasticNet()
reg_model.fit(X, y)
reg_model.intercept_
reg_model.coef_
y_pred = reg_model.predict(X)
sonuc_tum_veri_detay_enet = np.sqrt(mean_squared_error(y, y_pred))
sonuc_tum_veri_detay_enet
sonuc_tum_veri_cv_detay_enet = np.sqrt(np.mean(-cross_val_score(reg_model, X, y, cv = 10, scoring = "neg_mean_squared_error")))
sonuc_tum_veri_cv_detay_enet
reg_model = ElasticNet()
reg_model.fit(X_train, y_train)
reg_model.intercept_
reg_model.coef_
y_pred = reg_model.predict(X_train)
sonuc_train_veri_detay_enet = np.sqrt(mean_squared_error(y_train, y_pred))
sonuc_train_veri_detay_enet
y_pred = reg_model.predict(X_test)
sonuc_test_veri_detay_enet = np.sqrt(mean_squared_error(y_test, y_pred))
sonuc_test_veri_detay_enet
sonuc_tt_veri_cv_detay_enet = np.sqrt(np.mean(-cross_val_score(reg_model, X_train, y_train, cv = 10, scoring = "neg_mean_squared_error")))
sonuc_tt_veri_cv_detay_enet
#Detaylı sonuçların çıktısı
detaysonuc_df = pd.DataFrame({"Tüm Veri Score":[sonuc_tum_veri_detay_slr,sonuc_tum_veri_detay_clr,sonuc_tum_veri_detay_ridge,sonuc_tum_veri_detay_lasso,sonuc_tum_veri_detay_enet],
                              "Tüm Veri CVScore":[sonuc_tum_veri_cv_detay_slr,sonuc_tum_veri_cv_detay_clr,sonuc_tum_veri_cv_detay_ridge,sonuc_tum_veri_cv_detay_lasso,sonuc_tum_veri_cv_detay_enet],
                              "Train Veri Sonuc":[sonuc_train_veri_detay_slr,sonuc_train_veri_detay_clr,sonuc_train_veri_detay_ridge,sonuc_train_veri_detay_lasso,sonuc_train_veri_detay_enet],
                              "Test Veri Sonuc":[sonuc_test_veri_detay_slr,sonuc_test_veri_detay_clr,sonuc_test_veri_detay_ridge,sonuc_test_veri_detay_lasso,sonuc_test_veri_detay_enet],                              
                              "TrainTest CVScore":[sonuc_tt_veri_cv_detay_slr,sonuc_tt_veri_cv_detay_clr,sonuc_tt_veri_cv_detay_ridge,sonuc_tt_veri_cv_detay_lasso,sonuc_tt_veri_cv_detay_enet]})
detaysonuc_df.index= ["SLR_DETAY", "CLR_DETAY","RID_DETAY","LAS_DETAY","ENE_DETAY"]
detaysonuc_df
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import LassoCV
from sklearn.linear_model import ElasticNetCV
alphas1 = np.random.randint(0,1000,100)
alphas2 = 10**np.linspace(10,-2,100)*0.5
alphas3 = np.linspace(0,1,1000)
ridge_cv = RidgeCV(alphas = alphas1, scoring = "neg_mean_squared_error", cv = 10, normalize = True)
ridge_cv.fit(X, y)
tum_alpha_ridge1 = ridge_cv.alpha_
tum_alpha_ridge1
ridge_cv.fit(X_train, y_train)
tt_alpha_ridge1 = ridge_cv.alpha_
tt_alpha_ridge1 
ridge_tuned = Ridge(alpha = tum_alpha_ridge1).fit(X, y)
y_pred = ridge_tuned.predict(X)
tum_tuned_ridge1 = np.sqrt(mean_squared_error(y, y_pred))
tum_tuned_ridge1
ridge_tuned = Ridge(alpha = tt_alpha_ridge1 ).fit(X_train, y_train)
y_pred = ridge_tuned.predict(X_test)
tt_tuned_ridge1 = np.sqrt(mean_squared_error(y_test, y_pred))
tt_tuned_ridge1
ridge_cv = RidgeCV(alphas = alphas2, scoring = "neg_mean_squared_error", cv = 10, normalize = True)
ridge_cv.fit(X, y)
tum_alpha_ridge2 = ridge_cv.alpha_
tum_alpha_ridge2
ridge_cv.fit(X_train, y_train)
tt_alpha_ridge2 = ridge_cv.alpha_
tt_alpha_ridge2
ridge_tuned = Ridge(alpha = tum_alpha_ridge2).fit(X, y)
y_pred = ridge_tuned.predict(X)
tum_tuned_ridge2 = np.sqrt(mean_squared_error(y, y_pred))
tum_tuned_ridge2
ridge_tuned = Ridge(alpha = tt_alpha_ridge2).fit(X_train, y_train)
y_pred = ridge_tuned.predict(X_test)
tt_tuned_ridge2 = np.sqrt(mean_squared_error(y_test, y_pred))
tt_tuned_ridge2
ridge_cv = RidgeCV(alphas = alphas3, scoring = "neg_mean_squared_error", cv = 10, normalize = True)
ridge_cv.fit(X, y)
tum_alpha_ridge3 = ridge_cv.alpha_
tum_alpha_ridge3
ridge_cv.fit(X_train, y_train)
tt_alpha_ridge3 = ridge_cv.alpha_
tt_alpha_ridge3
ridge_tuned = Ridge(alpha = tum_alpha_ridge3).fit(X, y)
y_pred = ridge_tuned.predict(X)
tum_tuned_ridge3 = np.sqrt(mean_squared_error(y, y_pred))
tum_tuned_ridge3
ridge_tuned = Ridge(alpha = tt_alpha_ridge3).fit(X_train, y_train)
y_pred = ridge_tuned.predict(X_test)
tt_tuned_ridge3 = np.sqrt(mean_squared_error(y_test, y_pred))
tt_tuned_ridge3
if (tum_tuned_ridge1 < tum_tuned_ridge2) & (tum_tuned_ridge1 < tum_tuned_ridge3):
    tum_tuned_ridge = tum_tuned_ridge1
    tum_alpha_ridge = tum_alpha_ridge1
elif tum_tuned_ridge2 < tum_tuned_ridge3:
    tum_tuned_ridge = tum_tuned_ridge2
    tum_alpha_ridge = tum_alpha_ridge2
else:
    tum_tuned_ridge = tum_tuned_ridge3
    tum_alpha_ridge = tum_alpha_ridge3
print("RMSE:"+str(tum_tuned_ridge)+"  Alpha:"+str(tum_alpha_ridge))
if (tt_tuned_ridge1 < tt_tuned_ridge2) & (tt_tuned_ridge1 < tt_tuned_ridge3):
    tt_tuned_ridge = tt_tuned_ridge1
    tt_alpha_ridge = tt_alpha_ridge1
elif tum_tuned_ridge2 < tum_tuned_ridge3:
    tt_tuned_ridge = tt_tuned_ridge2
    tt_alpha_ridge = tt_alpha_ridge2
else:
    tt_tuned_ridge = tt_tuned_ridge3
    tt_alpha_ridge = tt_alpha_ridge3
print("RMSE:"+str(tt_tuned_ridge)+"  Alpha:"+str(tt_alpha_ridge))
alphas1 = np.random.randint(0,1000,100)
alphas2 = 10**np.linspace(10,-2,100)*0.5
alphas3 = np.linspace(0,1,1000)
lasso_cv = LassoCV(alphas = alphas1, cv = 10)
lasso_cv.fit(X, y)
tum_alpha_lasso1 = lasso_cv.alpha_
tum_alpha_lasso1
lasso_cv.fit(X_train, y_train)
tt_alpha_lasso1 = lasso_cv.alpha_
tt_alpha_lasso1
lasso_tuned = Lasso(alpha = tum_alpha_lasso1).fit(X, y)
y_pred = lasso_tuned.predict(X)
tum_tuned_lasso1 = np.sqrt(mean_squared_error(y, y_pred))
tum_tuned_lasso1
lasso_tuned = Lasso(alpha = tt_alpha_lasso1).fit(X_train, y_train)
y_pred = lasso_tuned.predict(X_test)
tt_tuned_lasso1 = np.sqrt(mean_squared_error(y_test, y_pred))
tt_tuned_lasso1
lasso_cv = LassoCV(alphas = alphas2, cv = 10)
lasso_cv.fit(X, y)
tum_alpha_lasso2 = lasso_cv.alpha_
tum_alpha_lasso2
lasso_cv.fit(X_train, y_train)
tt_alpha_lasso2 = lasso_cv.alpha_
tt_alpha_lasso2
lasso_tuned = Lasso(alpha = tum_alpha_lasso2).fit(X, y)
y_pred = lasso_tuned.predict(X)
tum_tuned_lasso2 = np.sqrt(mean_squared_error(y, y_pred))
tum_tuned_lasso2
lasso_tuned = Lasso(alpha = tt_alpha_lasso2).fit(X_train, y_train)
y_pred = lasso_tuned.predict(X_test)
tt_tuned_lasso2 = np.sqrt(mean_squared_error(y_test, y_pred))
tt_tuned_lasso2
lasso_cv = LassoCV(alphas = alphas3, cv = 10)
lasso_cv.fit(X, y)
tum_alpha_lasso3 = lasso_cv.alpha_
tum_alpha_lasso3
lasso_cv.fit(X_train, y_train)
tt_alpha_lasso3 = lasso_cv.alpha_
tt_alpha_lasso3
lasso_tuned = Lasso(alpha = tt_alpha_lasso3).fit(X, y)
y_pred = lasso_tuned.predict(X)
tum_tuned_lasso3 = np.sqrt(mean_squared_error(y, y_pred))
tum_tuned_lasso3
lasso_tuned = Lasso(alpha = tt_alpha_lasso3).fit(X_train, y_train)
y_pred = lasso_tuned.predict(X_test)
tt_tuned_lasso3 = np.sqrt(mean_squared_error(y_test, y_pred))
tt_tuned_lasso3
if (tum_tuned_lasso1 < tum_tuned_lasso2) & (tum_tuned_lasso1 < tum_tuned_lasso3):
    tum_tuned_lasso = tum_tuned_lasso1
    tum_alpha_lasso = tum_alpha_lasso1
elif tum_tuned_lasso2 < tum_tuned_lasso3:
    tum_tuned_lasso = tum_tuned_lasso2
    tum_alpha_lasso = tum_alpha_lasso2
else:
    tum_tuned_lasso = tum_tuned_lasso3
    tum_alpha_lasso = tum_alpha_lasso3
print("RMSE:"+str(tum_tuned_lasso)+"  Alpha:"+str(tum_alpha_lasso))
if (tt_tuned_lasso1 < tt_tuned_lasso2) & (tt_tuned_lasso1 < tt_tuned_lasso3):
    tt_tuned_lasso = tt_tuned_lasso1
    tt_alpha_lasso = tt_alpha_lasso1
elif tt_tuned_lasso2 < tt_tuned_lasso3:
    tt_tuned_lasso = tt_tuned_lasso2
    tt_alpha_lasso = tt_alpha_lasso2
else:
    tt_tuned_lasso = tt_tuned_lasso3
    tt_alpha_lasso = tt_alpha_lasso3
print("RMSE:"+str(tt_tuned_lasso)+"  Alpha:"+str(tt_alpha_lasso))
from sklearn.model_selection import GridSearchCV
enet_params = {"l1_ratio": [0.1,0.2,0.4,0.5,0.6,0.8,1],
              "alpha":[0.1,0.01,0.001,0.2,0.3,0.5,0.8,0.9,1]}
enet_model = ElasticNet()
gs_cv_enet = GridSearchCV(enet_model, enet_params, cv = 10).fit(X, y)
gs_cv_enet.best_params_
tum_tuned_enet_a = gs_cv_enet.best_params_["alpha"]
tum_tuned_enet_a
tum_tuned_enet_l = gs_cv_enet.best_params_["l1_ratio"]
tum_tuned_enet_l
enet_tuned = ElasticNet(**gs_cv_enet.best_params_).fit(X, y)
y_pred = enet_tuned.predict(X)
tum_tuned_enet = np.sqrt(mean_squared_error(y, y_pred))
tum_tuned_enet 
gs_cv_enet = GridSearchCV(enet_model, enet_params, cv = 10).fit(X_train, y_train)
gs_cv_enet.best_params_
tt_tuned_enet_a = gs_cv_enet.best_params_["alpha"]
tt_tuned_enet_a
tt_tuned_enet_l = gs_cv_enet.best_params_["l1_ratio"]
tt_tuned_enet_l
enet_tuned = ElasticNet(**gs_cv_enet.best_params_).fit(X_train, y_train)
y_pred = enet_tuned.predict(X_test)
tt_tuned_enet = np.sqrt(mean_squared_error(y_test, y_pred))
tt_tuned_enet 
#Detaylı sonuçların çıktısı
detaysonuc_df = pd.DataFrame({"All Data Error":[sonuc_tum_veri_detay_slr,sonuc_tum_veri_detay_clr,sonuc_tum_veri_detay_ridge,sonuc_tum_veri_detay_lasso,sonuc_tum_veri_detay_enet],
                              "All Data cvScore(K-Fold)":[sonuc_tum_veri_cv_detay_slr,sonuc_tum_veri_cv_detay_clr,sonuc_tum_veri_cv_detay_ridge,sonuc_tum_veri_cv_detay_lasso,sonuc_tum_veri_cv_detay_enet],
                              "All Data Tuned Error":[np.nan,np.nan,tum_tuned_ridge,tum_tuned_lasso,tum_tuned_enet],
                              "Train Error":[sonuc_train_veri_detay_slr,sonuc_train_veri_detay_clr,sonuc_train_veri_detay_ridge,sonuc_train_veri_detay_lasso,sonuc_train_veri_detay_enet],
                              "Test Error":[sonuc_test_veri_detay_slr,sonuc_test_veri_detay_clr,sonuc_test_veri_detay_ridge,sonuc_test_veri_detay_lasso,sonuc_test_veri_detay_enet],                              
                              "Train cvScore(K-Fold)":[sonuc_tt_veri_cv_detay_slr,sonuc_tt_veri_cv_detay_clr,sonuc_tt_veri_cv_detay_ridge,sonuc_tt_veri_cv_detay_lasso,sonuc_tt_veri_cv_detay_enet],
                              "Test Tuned Error":[np.nan,np.nan,tt_tuned_ridge,tt_tuned_lasso,tt_tuned_enet],
                              "Alphas":[np.nan,np.nan,tt_alpha_ridge,tt_alpha_lasso,tt_tuned_enet_a]})
detaysonuc_df.index= ["SLR_DETAY", "CLR_DETAY","RID_DETAY","LAS_DETAY","ENE_DETAY"]
detaysonuc_df
basicsonuc_df