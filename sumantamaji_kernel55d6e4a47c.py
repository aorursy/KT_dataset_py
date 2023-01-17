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
df=pd.read_csv("/kaggle/input/quikr_car.csv")
df
df['kms_driven'] = df['kms_driven'].str.replace(',', '')# replace the commas with nothing


df.isnull().sum()
df['kms_driven'].fillna("123",inplace=True)
driven=list(df['kms_driven'])#converted the kms_driven data in list
km_driven=[]
for i in range(len(driven)):
    #print(driven[i])
    print(type(driven[i]))
    print(i)
    km_driven.append(int(driven[i].split(sep = " ")[0]))
    print(type(km_driven[i]))

km_driven
df.drop(["kms_driven"], axis = 1, inplace = True)
df["kms_driven"] = km_driven
df
df['kms_driven'].replace(123,0,inplace=True)
df
df['Price'].isnull().sum()#no nule inside Price column
df[df['Price']=="Ask For Price"]
df[df['name']=="Maruti Ertiga showroom condition with"]

df[df['company']=="Ford"].groupby("name").mean().head(10)
type(df['Price'][0])
df['Price'].replace("Ask For Price","1",inplace=True)
df['Price'] = df['Price'].str.replace(',', '')
df
price=list(df['Price'])#converted the kms_driven data in list
p=[]
for i in range(len(price)):
    #print(driven[i])
    print(type(price[i]))
    print(i)
    p.append(int(price[i]))
    print(type(p[i]))

df.drop(["Price"], axis = 1, inplace = True)
df["Price"]=p
import numpy as np
df['Price'].replace(1,np.NaN,inplace=True)
df
df[df['Price'].isnull()]
df[df['name']=="Maruti Suzuki Swift LDi"]
df[df['name']=="Maruti Suzuki Swift LDi"].mean()
df.loc[295,"Price"]=272500.000000
df[df['name']=="Maruti Suzuki Alto 800 Vxi"]
df[df['name']=="Maruti Suzuki Alto 800 Vxi"].mean()

df.loc[1,"Price"]= 175000.0
df[df['name']=="Ford EcoSport Titanium 1.5L TDCi"]
df[df['name']=="Ford EcoSport Titanium 1.5L TDCi"].mean()
df.loc[1,"Price"]= 556000.0
df[df['name']=="I want to sell my car Tata Zest"]
df.drop([69,85], axis=0, inplace = True)
df[df['name']=="Maruti Suzuki Alto 800 Lxi"]
df[df['name']=="Maruti Suzuki Alto 800 Lxi"].mean()
df.loc[[138,388],"Price"]=257857.142857
df[df['name']=="Commercial , DZire LDI, 2016, for sale"]
df.drop([185,286], axis=0, inplace = True)
df[df['name']=="Tata Indica eV2 LS"]
df[df['name']=="Tata Indica eV2 LS"].mean()
df.loc[[304],"Price"]=95250.0
df[df['name']=="selling car Ta"]
df.drop([360], axis=0, inplace = True)
df[df['name']=="Tata Zest 90"]
df.drop([368], axis=0, inplace = True)
df[df['name']=="Maruti Suzuki Zen Estilo LXI Green CNG"]
df[df['name']=="Maruti Suzuki Zen Estilo LXI Green CNG"].mean()
df.loc[449,"Price"]=128333.333333
df[df['name']=="Hyundai Xcent Base 1.1 CRDi"]
df[df['name']=="Hyundai Xcent Base 1.1 CRDi"].mean()
df.loc[503,"Price"]=300000.0
df[df['name']=="Hyundai Xcent S 1.2"]
df[df['name']=="Toyota Innova 2.0 V"]
df[df['name']=="Hyun"]
df.drop([560], axis=0, inplace = True)
df[df['name']=="Datsun Go Plus T O"]
df.loc[567,"Price"]=237000.0000
df[df['name']=="Mahindra KUV100 K8 D 6 STR"].mean()
df.loc[613,"Price"]=560000.0000
df[df['name']=="Maruti Suzuki Alto LX BSII"]
df.loc[619,"Price"]=55005.0000
df[df['name']=="Hyundai Elite i20 Sportz 1.2"]
df.loc[634,"Price"]=400000.0000

df[df['name']=="Sale Hyundai xcent commerc"]
df.drop([645], axis=0, inplace = True)

df[df['name']=="Tata Nexon"]
df.loc[763,"Price"]=670000.0000
df[df['name']=="Tata"]
df.drop([764], axis=0, inplace = True)
df[df['name']=="7 SEATER MAHINDRA BOLERO IN VERY GOOD"]
df.drop([798], axis=0, inplace = True)
df[df['name']=="9 SEATER MAHINDRA BOL"]
df.drop([799], axis=0, inplace = True)
df[df['name']=="Hyunda"]
df.drop([808,811], axis=0, inplace = True)
df[df['Price'].isnull()]
df[df['name']=="Maruti Suzuki Alto 800 Vxi"]
df.loc[2,"Price"]=230000.0000
df[df['name']=="Maruti Suzuki Alto 800 Select Variant"]
df.loc[882,"Price"]=200000.0000
df[df['name']=="Toyota Innova 2.0 G1 Petrol 8seater"]
df.loc[859,"Price"]=930000.0000
df[df['name']=="Volkswagen Vento Highline Plus 1.5 Diesel"]
df.loc[854,"Price"]=380000.0000 
df[df['name']=="Ford EcoSport Titanium 1.5L TDCi"].mean()
df.loc[5,"Price"]=556000.0
df[df['name']=="Hyundai Xcent S 1.2"]
df.loc[511,"Price"]=430000.0
df[df['name']=="Toyota Innova 2.0 V"]
df.loc[524,"Price"]=630000.0
df[df['name']=="Hyunda"]
df.drop([807], axis=0, inplace = True)
df[df['name']=="Hyundai Venue"]
df.loc[524,"Price"]=550000.0
df[df['name']=="Renault Lodgy"]
df.loc[524,"Price"]=530000.0
df[df['name']=="Maruti Suzuki Alto 800 Lx"]
df.loc[848,"Price"]=400000.0
df[df['name']=="Maruti Suzuki Alto 800 Lx"]
df
df[df['fuel_type'].isnull()]
df[df['name']=="Tata indigo ecs LX, 201"]

df[df['fuel_type'].isnull()]
df["fuel_type"].fillna( method ='ffill', inplace = True)

df[df['name']=="Toyota Corolla"]
df[df['year'].isnull()]
df["Price"].isnull().sum()
df.head(30)
df['fuel_type'].value_counts()
df['fuel_type']=df['fuel_type'].replace({'Petrol':0,'Diesel':1,'LPG':2})
df
X=df_train.drop(["Price"],axis=1)
y=df_train.loc[:,"Price"]

from sklearn.preprocessing import OneHotEncoder,PolynomialFeatures
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score,make_scorer
from sklearn.model_selection import cross_val_score,GridSearchCV
from sklearn.ensemble import RandomForestRegressor
lr= RandomForestRegressor()
ohe=OneHotEncoder()
ohe.fit(X[['name','company','fuel_type']])
column_trans=make_column_transformer((OneHotEncoder(categories=ohe.categories_),['name','company','fuel_type']),
                                    (PolynomialFeatures(degree=20),['kms_driven']),remainder='passthrough')
pipe=make_pipeline(column_trans,lr)
scoring_func=make_scorer(r2_score,greater_is_better=False)
-cross_val_score(pipe,X,y,scoring=scoring_func,cv=10)


pipe.fit(X_train,y_train)
y_pred=pipe.predict(X_test)
r2_score(y_test,y_pred)
scores=[]
for i in range(1000):
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.1,random_state=i)
    lr=LinearRegression()
    pipe=make_pipeline(column_trans,lr)
    pipe.fit(X_train,y_train)
    y_pred=pipe.predict(X_test)
    scores.append(r2_score(y_test,y_pred))
np.argmax(scores)

scores[np.argmax(scores)]
pipe.predict(pd.DataFrame(columns=X_test.columns,data=np.array(['Maruti Suzuki Swift','Maruti',2019,100,'Petrol']).reshape(1,5)))
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.1,random_state=np.argmax(scores))
lr=LinearRegression()
pipe=make_pipeline(column_trans,lr)
pipe.fit(X_train,y_train)
y_pred=pipe.predict(X_test)
r2_score(y_test,y_pred)
df['year'].value_counts()
# As name is Nominal Categorical data we will perform OneHotEncoding

name= df[["name"]]

name = pd.get_dummies(name, drop_first= True)

name.head()
company= df[["company"]]

company = pd.get_dummies(company, drop_first= True)

company.head()




df_train = pd.concat([df,name, company, year], axis = 1)
df_train
df.iloc[:,0]


df_train.drop(["name", "company", "year"], axis = 1, inplace = True)
df_train

df_train_list=list()
df_train_list=df_train.columns

len(df_train_list)
df_train.groupby("name_Audi A6 2.0 TDI Premium").sum()
df_train_list[0:10]
df_train.dropna(inplace = True)
X=df_train.drop(["Price"],axis=1)
y=df_train.loc[:,"Price"]

from sklearn import linear_model
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=0)
reg = linear_model.LinearRegression()
reg.fit(X_train,y_train)

y_pred = reg.predict(X_test)
y_pred

reg.score(X_train, y_train)
reg.score(X_test, y_test)

from sklearn.ensemble import RandomForestRegressor
reg_rf = RandomForestRegressor()
reg_rf.fit(X_train, y_train)
y_pred = reg_rf.predict(X_test)


reg_rf.score(X_train, y_train)

reg_rf.score(X_test, y_test)
from sklearn import metrics

metrics.r2_score(y_test, y_pred)
import xgboost as xgb
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_squared_error
xgbr = xgb.XGBRegressor(verbosity=0)


xgbr.fit(X_train, y_train)
y_pred = xgbr.predict(X_test)
xgbr.score(X_train, y_train)
xgbr.score(X_test, y_test)
from sklearn import metrics

metrics.r2_score(y_test, y_pred)
scores = cross_val_score(xgbr,X_train ,y_train , cv=5)
print("Mean cross-validation score: %.2f" % scores.mean())

kfold = KFold(n_splits=10, shuffle=True)
kf_cv_scores = cross_val_score(xgbr,X_train , y_train, cv=kfold )
print("K-fold CV average score: %.2f" % kf_cv_scores.mean())
 
ypred = xgbr.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print("MSE: %.2f" % mse)
print("RMSE: %.2f" % (mse*(1/2.0)))

## Hyper Parameter Optimization



# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 100, stop = 500, num = 5)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 30, num = 3)]
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10, 15]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4,6]

random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf}
## Hyperparameter optimization using RandomizedSearchCV
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
import xgboost
rf_random = RandomizedSearchCV(estimator = reg_rf, param_distributions = random_grid,scoring='neg_mean_squared_error', n_iter = 10, cv = 5, verbose=2, random_state=42, n_jobs = 1)
rf_random.fit(X_train,y_train)


rf_random.best_params_
prediction = rf_random.predict(X_test)
metrics.r2_score(y_test, prediction)

import pickle
# open a file, where you ant to store the data
file = open('car_price.pkl', 'wb')

# dump information to that file
pickle.dump(xgbr, file)
