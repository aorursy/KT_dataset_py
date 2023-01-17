# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.svm import SVC

import xgboost as xgb

from sklearn import metrics

from sklearn.model_selection import RandomizedSearchCV

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df=pd.read_csv('/kaggle/input/vehicle-dataset-from-cardekho/CAR DETAILS FROM CAR DEKHO.csv')
df.head()
df.dtypes
df.shape
df.isna().sum()
df.drop('name',axis=1,inplace=True)
df_encoded=pd.get_dummies(df,drop_first=True)
df_encoded
df_encoded.corr()
corrmat = df_encoded.corr()

top_corr_features = corrmat.index

fig, ax = plt.subplots(figsize=(10,10))

g=sns.heatmap(df_encoded[top_corr_features].corr(),annot=True,cmap="RdYlGn")
df_encoded.columns
X=df_encoded[['year', 'km_driven', 'fuel_Diesel', 'fuel_Electric',

       'fuel_LPG', 'fuel_Petrol', 'seller_type_Individual',

       'seller_type_Trustmark Dealer', 'transmission_Manual',

       'owner_Fourth & Above Owner', 'owner_Second Owner',

       'owner_Test Drive Car', 'owner_Third Owner']]

y=df_encoded['selling_price']
X.shape
y.shape
X.head()
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20,random_state=69)
xgbr=xgb.XGBRegressor().fit(X_train,y_train)

y_pred=xgbr.predict(X_test)
sns.distplot(y_test-y_pred)
sns.regplot(y_test,y_pred)
print('MAE:', metrics.mean_absolute_error(y_test, y_pred))

print('MSE:', metrics.mean_squared_error(y_test, y_pred))

print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

print('mean of selling price', df['selling_price'].mean())
param_grid={'learning_rate':[0.1,0.25,0.5],

             'n_estimators':[100,500,1000],

             'max_depth':[3,5,7],

             'gamma':[0,0.1,0.05,0.001],

           }

rs_xgb=RandomizedSearchCV(xgb.XGBRegressor(objective='reg:squarederror',n_jobs=4),param_grid,cv=5)

rs_xgb.fit(X_train,y_train)
y_pred1=rs_xgb.predict(X_test)

print('MAE:', metrics.mean_absolute_error(y_test, y_pred1))

print('MSE:', metrics.mean_squared_error(y_test, y_pred1))

print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_pred1)))