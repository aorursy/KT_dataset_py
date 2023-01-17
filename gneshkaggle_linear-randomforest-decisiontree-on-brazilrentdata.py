# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.

import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.simplefilter(action='ignore')

df=pd.read_csv('/kaggle/input/brasilian-houses-to-rent/houses_to_rent_v2.csv')
df.head()


df.shape
df.city.unique()
df=df.rename(columns={'hoa (R$)':'hoa','rent amount (R$)':'rentamount','property tax (R$)':'propertytax',
                   'fire insurance (R$)':'fireinsurance','total (R$)':'total'})
df.head()
corr=df.corr()
plt.rcParams['figure.figsize']=(12,8)
sns.heatmap(corr,annot=True,fmt='.2f')
#dropping outlier indices
df.drop([2397,5915,9241],inplace=True)
df.floor.replace(to_replace='-',value='0',inplace=True)
df.floor=df.floor.astype('int64')
df.floor.unique()
df.drop(['floor'],axis=1,inplace=True)
sns.distplot(df.total)
#checking for outliers
df.total.sort_values(ascending=False)
#removing rows which have 'total' outliers
df.drop([255,6979,6645,6230,2859,2928,2182],inplace=True)
sns.distplot(df.total)
sns.countplot(df.rooms)
sns.pairplot(df)
sns.lmplot(x='area',y='total',hue='furniture',data=df,fit_reg=False)
sns.boxplot(x='furniture',y='total',data=df)
sns.barplot(x='furniture',y='total',data=df)
df.furniture=df.furniture.map({'furnished':1,'not furnished':0})
sns.boxplot(x='animal',y='total',data=df)
df.animal=df.animal.map({'acept':1,'not acept':0})
df.city=df.city.map({'São Paulo':1,'Porto Alegre':2,'Rio de Janeiro':3,
                      'Campinas':4,'Belo Horizonte':5})
df.head()
X=df.loc[:,df.columns!='total']
y=df.loc[:,'total']
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=12)
from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(X_train,y_train)
#linear regression using data which includes 'hoa','propertytax,'fireinsurance
reg.score(X_train,y_train)
reg_pred=reg.predict(X_test)
reg.score(X_test,y_test)
X_train.head()
#dropping the columns other than total column
X_train.drop(['hoa','rentamount','propertytax','fireinsurance'],axis=1,inplace=True)
X_test.drop(['hoa','rentamount','propertytax','fireinsurance'],axis=1,inplace=True)
reg.coef_
#linear regression on new data
regg=LinearRegression()
regg.fit(X_train,y_train)
regg.score(X_train,y_train)
regg.coef_
regg_pred=regg.predict(X_test)
from sklearn.metrics import mean_squared_error

#printing the error without using data containing columns 'hoa','rentamount','propertytax','fireinsurance'
print('train error ',mean_squared_error(y_test,regg_pred))
#printing the error using data containing columns 'hoa','rentamount','propertytax','fireinsurance'

print('train error ',mean_squared_error(y_test,reg_pred))
X_train.head()
from sklearn.ensemble import RandomForestRegressor
rf=RandomForestRegressor()
rf.fit(X_train,y_train)

rf.score(X_train,y_train)
rf_pred=rf.predict(X_test)
print('MSE:',mean_squared_error(rf_pred,y_test))
#GridSearchCV on random forest regressor
from sklearn.model_selection import GridSearchCV,cross_val_score
params={'n_estimators':[10,25,50,100,200],'max_depth':np.arange(2,10),
       'min_samples_leaf':np.arange(2,5)}
rf_best=GridSearchCV(rf,param_grid=params,verbose=1,n_jobs=-1)
rf_best.fit(X_train,y_train)
rf_best.best_params_,rf_best.best_score_
rf_best.score(X_train,y_train)
rf_best_pred=rf_best.predict(X_test)
print("MSE:",mean_squared_error(rf_best_pred,y_test))
from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor()
dt.fit(X_train,y_train)
dt.score(X_train,y_train)
dt_pred=dt.predict(X_test)
print('MSE:',mean_squared_error(dt_pred,y_test))
dt.get_params()
dt_params={'max_depth':np.arange(2,10),'min_samples_leaf':np.arange(2,8)}
dt_best=GridSearchCV(dt,param_grid=dt_params,n_jobs=-1)
dt_best.fit(X_train,y_train)
dt_best.best_params_,dt_best.best_score_
dt_best_pred=dt_best.predict(X_test)
print("MSE:",mean_squared_error(dt_best_pred,y_test))
from sklearn.metrics import r2_score
print("MSE:",r2_score(dt_best_pred,y_test))
df.head(2)
columns=['city','area','rooms','bathroom','parking spaces','animal','furniture']
xx=df.loc[:,columns]
yy=df.loc[:,'rentamount']
xx_train,xx_test,yy_train,yy_test=train_test_split(xx,yy,test_size=0.3,random_state=123)
lrc=LinearRegression()
lrc.fit(xx_train,yy_train)
lrc.score(xx_train,yy_train)
lrc_pred=lrc.predict(xx_test)
print('r2 score:',r2_score(lrc_pred,yy_test))
print('MSE:',mean_squared_error(lrc_pred,yy_test))
#for improving accuracy:
#trying back elimination algorithm would increase the accuracy
#removing outliers for all the columns
