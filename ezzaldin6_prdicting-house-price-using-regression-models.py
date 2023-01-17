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
df=pd.read_csv('/kaggle/input/housesalesprediction/kc_house_data.csv')

df.head(2)
df.info()
useless_cols=['id','date','lat','long','zipcode']

df=df.drop(useless_cols,axis=1)
float_int_lst=['bathrooms','floors']

df[float_int_lst]=df[float_int_lst].astype('int')
df.corr()
import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

sns.heatmap(df.corr())
corr_abs_data=df.corr()['price'].abs().sort_values()

corr_abs_data
linear_rel_data=df.drop(corr_abs_data[(corr_abs_data<0.2)|(corr_abs_data==np.nan)].index,axis=1)

linear_rel_data.columns
from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error,r2_score

from sklearn.model_selection import train_test_split

def linear_reg(df,target):

    features=df.columns.drop(target)

    x_train,x_test,y_train,y_test=train_test_split(df[features],df[target],test_size=0.2,random_state=1)

    lr=LinearRegression()

    lr.fit(x_train,y_train)

    pred=lr.predict(x_test)

    mse=mean_squared_error(y_test,pred)

    rmse=np.sqrt(mse)

    r2=r2_score(y_test,pred)

    return(mse,rmse,r2)

linear_reg(df,'price')
from sklearn.preprocessing import PolynomialFeatures

def polynomial_reg(df,target,power):

    features=df.columns.drop(target)

    x_train,x_test,y_train,y_test=train_test_split(df[features],df[target],test_size=0.2,random_state=1)

    poly=PolynomialFeatures(degree=power)

    x_train=poly.fit_transform(x_train)

    x_test=poly.fit_transform(x_test)

    poly_lr=LinearRegression()

    poly_lr.fit(x_train,y_train)

    pred=poly_lr.predict(x_test)

    mse=mean_squared_error(y_test,pred)

    rmse=np.sqrt(mse)

    r2=r2_score(y_test,pred)

    return(mse,rmse,r2)

polynomial_reg(df,'price',2)
from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestRegressor

def rf_reg(df,target,esti):

    features=df.columns.drop(target)

    x_train,x_test,y_train,y_test=train_test_split(df[features],df[target],test_size=0.2,random_state=1)

    rf=RandomForestRegressor(n_estimators=esti,random_state=0)

    rf.fit(x_train,y_train)

    pred=rf.predict(x_test)

    mse=mean_squared_error(y_test,pred)

    rmse=np.sqrt(mse)

    r2=r2_score(y_test,pred)

    return(mse,rmse,r2)

rf_reg(df,'price',30)
def dtree_reg(df,target):

    features=df.columns.drop(target)

    x_train,x_test,y_train,y_test=train_test_split(df[features],df[target],test_size=0.2,random_state=1)

    dtree=DecisionTreeRegressor(random_state=0)

    dtree.fit(x_train,y_train)

    pred=dtree.predict(x_test)

    mse=mean_squared_error(y_test,pred)

    rmse=np.sqrt(mse)

    r2=r2_score(y_test,pred)

    return(mse,rmse,r2)

rf_reg(df,'price',30)
from sklearn.neighbors import KNeighborsRegressor

def k_reg(df,target):

    features=df.columns.drop(target)

    x_train,x_test,y_train,y_test=train_test_split(df[features],df[target],test_size=0.2,random_state=1)

    acc_scores={}

    for i in range(1,15):

        knn=KNeighborsRegressor(n_neighbors=i)

        knn.fit(x_train,y_train)

        pred=knn.predict(x_test) 

        mse=mean_squared_error(y_test,pred)

        rmse=np.sqrt(mse)

        r2=r2_score(y_test,pred)

        acc_tuble=(mse,rmse,r2)

        acc_scores[i]=acc_tuble

    return acc_scores

k_reg(df,'price')