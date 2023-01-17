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
import numpy as np
import pandas as pd
import seaborn as sns
import io
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split, GridSearchCV,cross_val_score
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import metrics
from sklearn import neighbors
from sklearn.svm import SVR
from warnings import filterwarnings
filterwarnings('ignore')
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression, Lasso, ElasticNet, Ridge
from sklearn import model_selection 
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
#google colab üzerinde çalışanlar bu şekilde veri setini içeriye aktarabilir
#uploaded = files.upload()        
#df=pd.read_csv(io.BytesIO(uploaded['turkey_car_market.csv']))
df=pd.read_csv("/kaggle/input/turkey-car-market-2020/turkey_car_market.csv")
df.shape
df.head()
df.isnull().sum()
df.info()
sns.catplot(x="Yakıt Turu", y="Fiyat", kind="bar", data=df);
sns.catplot(x="Vites", y="Fiyat", kind="bar", data=df, palette="ch:.25");
sns.catplot(x="Durum", y="Fiyat", kind="bar", data=df);
sns.catplot(x="Kimden", y="Fiyat", kind="bar", data=df, palette="ch:.25");
print(len(df['Marka'].unique()))               
markalar=df['Marka'].unique()
print(markalar)
print(len(df['Arac Tip'].unique()))
car_type=df['Arac Tip'].unique()
print(car_type)
len(df[df['Arac Tip']=='-'])  
print(len(df['Yakıt Turu'].unique()))
yakıt_type=df['Yakıt Turu'].unique()
yakıt_type
print(len(df['Vites'].unique()))
vites_type=df['Vites'].unique()
vites_type
print('Farklı CCM değer sayısı : ', len(df['CCM'].unique()),'\n')
CCM_type=df['CCM'].unique()
print(CCM_type,'\n')
print('Bilmiyorum değeri girilmiş CCM sayısı : ', len(df[df['CCM']=='Bilmiyorum']))
print('Farklı Beygir gucu değer sayısı : ', len(df['Beygir Gucu'].unique()),'\n')
power=df['Beygir Gucu'].unique()
print(power,'\n')
print('Bilmiyorum değeri girilmiş Beygir Gucu sayısı : ', len(df[df['Beygir Gucu']=='Bilmiyorum']))
df['car_age']=2020-df['Model Yıl']     #modelin yılı yerine yaşı ile işlem yapıcağız
df['car_age'].head()
df.drop(['Model Yıl'], axis=1,inplace=True)    #işimize yaramayan kolonları siliyoruz
df.drop(['İlan Tarihi'],axis=1,inplace=True)
df.columns
df['Arac Tip']=df['Arac Tip'].str.replace('-','Diger')
len(df[df['Arac Tip']=='Diger'])
CCM_drop=df[df['CCM']=='Bilmiyorum'].index
df.drop(CCM_drop,axis=0,inplace=True)
df.shape
l_encoder1=LabelEncoder()
df['Marka']=l_encoder1.fit_transform(df['Marka'])
cars={}
car_name = list(l_encoder1.inverse_transform([i for i in range(35)]))
for i,x in enumerate(car_name):
  if i not in cars.keys():
    cars[i] =x
pd.DataFrame(cars.items(), columns=['label_values', 'car_name']).head()
l_encoder = LabelEncoder()
columns = ['Arac Tip Grubu', 'Arac Tip','Yakıt Turu', 'Vites', 'CCM', 'Beygir Gucu', 'Renk', 'Kasa Tipi','Kimden', 'Durum']
for i in columns:
  df[i]=l_encoder.fit_transform(df[i])
df.head()
df.info()
df['Beygir Gucu']=df['Beygir Gucu'].replace(18,np.nan)  
df.isnull().sum()
imputer = KNNImputer(n_neighbors=5)
df['Beygir Gucu']=imputer.fit_transform(df[['Beygir Gucu']])
df.isnull().sum()
df['Beygir Gucu'] = round(df['Beygir Gucu'])   #doldururken float olarak bıraktığı için tam değere yuvarlıyorum
q1 = df["Fiyat"].quantile(0.25)
q3 = df["Fiyat"].quantile(0.75)      

IOC = q3 - q1

alt_sınır = q1 - 1.5*IOC
üst_sınır = q3 + 1.5*IOC

sınır = (df["Fiyat"] < alt_sınır) | (df["Fiyat"] > üst_sınır)
df["Aykırı_Deger"] = sınır
print('Aykırı Değer Sayısı =>\n',df["Aykırı_Deger"].value_counts())
df = df.loc[df["Aykırı_Deger"] == False]
del df["Aykırı_Deger"]
y=df['Fiyat']
x=df.drop(['Fiyat'],axis=1)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.20, random_state = 42)
x_train.head()
sc_X = StandardScaler()
x_train = sc_X.fit_transform(x_train)
x_test = sc_X.transform(x_test)
modeller=[]
scores=[]
def compML(alg,x_train,y_train,x_test,y_test):
    model=alg().fit(x_train,y_train)
    y_pred=model.predict(x_test)
    RMSE= np.sqrt(mean_squared_error(y_test,y_pred))
    model_ismi= alg.__name__
    model_score = model.score(x_test,y_test)
    scores.append(model_score*100 if model_score > 0 else 0)
    modeller.append(model_ismi)
    print(model_ismi ," Modeli Test Hatası => ", RMSE,' |  Model Score => ', model_score*100)

models=[LGBMRegressor, Lasso,
        XGBRegressor, LinearRegression,
        GradientBoostingRegressor,
        RandomForestRegressor, ElasticNet,
        DecisionTreeRegressor, Ridge,
        MLPRegressor,
        KNeighborsRegressor, 
        SVR]

for i in models:
    compML(i,x_train,y_train,x_test,y_test) 
plt.figure(figsize=(15,10))
ax = sns.barplot(x=scores, y=modeller, palette="ch:4.5,-.7,dark=.3")
ax.set_title("Model-Skor Tablosu")
ax.set_ylabel("Modeller")
ax.set_ylabel("Score")
plt.show()
lgbm=LGBMRegressor()     
lgbm.fit(x_train,y_train)

lgbm_pred = lgbm.predict(x_test)

model_score = lgbm.score(x_test,y_test)
r2_skor = r2_score(y_test, lgbm_pred)
hata_skor = np.sqrt(mean_squared_error(y_test, lgbm_pred))
ev = metrics.explained_variance_score(y_test, lgbm_pred)

print("Model Score: ", model_score*100)
print("R2_skoru: ", r2_skor)
print("Hata Kare: ", hata_skor)
print("Explained Variance : ", ev)
lgbm_params={'learning_rate':[0.01, 0.1, 0.5],   
            'n_estimators':[200,500,1000],
            'max_depth':[5, 7, 10],
             'colsample_bytree':[0.7, 0.9, 1.0],
             'subsample': [0.4, 0.5, 0.6, 0.7]
             }
lgbm_cv_model=GridSearchCV(lgbm,lgbm_params,cv=10,n_jobs=-1,verbose=2).fit(x_train,y_train)
lgbm_cv_model.best_params_
lgbm_tuned=LGBMRegressor(learning_rate = lgbm_cv_model.best_params_['learning_rate'],
                        max_depth = lgbm_cv_model.best_params_['max_depth'],
                        n_estimators = lgbm_cv_model.best_params_['n_estimators'],
                        colsample_bytree = lgbm_cv_model.best_params_['colsample_bytree'],
                        subsample=lgbm_cv_model.best_params_['subsample']).fit(x_train,y_train)

y_pred=lgbm_tuned.predict(x_test)

model_score = lgbm_tuned.score(x_test,y_test)
r2_skor = r2_score(y_test, y_pred)
hata_skor = np.sqrt(mean_squared_error(y_test, y_pred))
adjusted_r2_skor = 1 - (1-r2_skor)*(len(y_test)-1)/(len(y_test)-x_test.shape[1]-1)
ev = metrics.explained_variance_score(y_test, y_pred)

print("Model scoru : ", model_score*100)
print("R2_skoru: ", r2_skor)
print("Hata Kare: ", hata_skor)
print("Adjusted_R2_skoru : ", adjusted_r2_skor)
print("Explained Variance : ", ev)
xgb=XGBRegressor()     
xgb.fit(x_train,y_train)

xgb_pred = xgb.predict(x_test)

model_score = xgb.score(x_test,y_test)
r2_skor = r2_score(y_test, xgb_pred)
hata_skor = np.sqrt(mean_squared_error(y_test, xgb_pred))
ev = metrics.explained_variance_score(y_test, xgb_pred)

print("Model Score: ", model_score*100)
print("R2_skoru: ", r2_skor)
print("Hata Kare: ", hata_skor)
print("Explained Variance : ", ev)
xgb_params = {'learning_rate':[0.01, 0.1, 0.5],
              'max_depth':[5, 7, 10],
             'colsample_bytree':[0.7, 0.9, 1.0],
             'subsample': [0.4, 0.5, 0.6, 0.7]}
xgb_cv_model=GridSearchCV(xgb,xgb_params,cv=10,n_jobs=-1,verbose=2).fit(x_train,y_train)
lgbm_cv_model.best_params_
xgb_tuned=XGBRegressor(learning_rate = xgb_cv_model.best_params_['learning_rate'],
                        max_depth = xgb_cv_model.best_params_['max_depth'],
                        colsample_bytree = xgb_cv_model.best_params_['colsample_bytree'],
                        subsample=xgb_cv_model.best_params_['subsample']).fit(x_train,y_train)

y_pred=xgb_tuned.predict(x_test)

model_score = xgb_tuned.score(x_test,y_test)
r2_skor = r2_score(y_test, y_pred)
hata_skor = np.sqrt(mean_squared_error(y_test, y_pred))
adjusted_r2_skor = 1 - (1-r2_skor)*(len(y_test)-1)/(len(y_test)-x_test.shape[1]-1)
ev = metrics.explained_variance_score(y_test, y_pred)

print("Model scoru : ", model_score*100)
print("R2_skoru: ", r2_skor)
print("Hata Kare: ", hata_skor)
print("Adjusted_R2_skoru : ", adjusted_r2_skor)
print("Explained Variance : ", ev)
sns.scatterplot(x=y_test,y=y_pred)
sns.jointplot(x=y_test, y=y_pred,  kind='reg',
                  joint_kws={'line_kws':{'color':'cyan'}})
real_pred = pd.DataFrame({'Gerçek Fiyat': np.array(y_test).flatten(), 'Tahmini Fiyat': y_pred.flatten(),'Fark':np.array(y_test).flatten()-y_pred.flatten()})
real_pred.Fark=round(real_pred.Fark)
real_pred.head(10)
