# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

#import os

import statsmodels.formula.api as sm

import sklearn.preprocessing as sk

# Any results you write to the current directory are saved as output.

from sklearn.metrics import mean_squared_error,mean_squared_log_error,r2_score

from sklearn.ensemble import GradientBoostingRegressor,RandomForestRegressor,AdaBoostRegressor

from sklearn.linear_model import LinearRegression

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import weight_boosting

from sklearn.ensemble import BaggingRegressor

from sklearn.neighbors import  KNeighborsRegressor

from sklearn.svm import SVR

from tensorflow import keras
train_data = pd.read_csv('../input/train.csv')

test_data = pd.read_csv('../input/test.csv')

y_train=train_data.pop('SalePrice')

test_id=test_data['Id'].values.reshape(-1,1)
y_train = np.log(y_train)

train_data.keys()
train_data.isna().sum()[train_data.isna().sum()>0]
train_data=train_data.drop(columns=['PoolQC','Fence','MiscFeature','Alley'],axis=0)

test_data=test_data.drop(columns=['PoolQC','Fence','MiscFeature','Alley'],axis=0)
cat_list = list(train_data.dtypes[train_data.dtypes == 'object'].index)
sts=train_data.describe().transpose()

train_data.fillna(sts['mean'],inplace=True)

test_data.fillna(sts['mean'],inplace=True)
train_data.isna().sum()[train_data.isna().sum()>0]
test_data.isna().sum()[test_data.isna().sum()>0]
for x in cat_list:

     if (train_data[x].isna().sum() > 0) :

        train_data[x].fillna(max(train_data[x].mode()), inplace=True)

     elif (test_data[x].isna().sum() > 0) :

        test_data[x].fillna(max(train_data[x].mode()), inplace=True)  
train_data.isna().sum()[train_data.isna().sum()>0]
test_data.isna().sum()[train_data.isna().sum()>0]
keys=train_data.keys()
train_data=pd.DataFrame(data=train_data,columns=keys)

test_data=pd.DataFrame(data=test_data,columns=keys)

total_data=pd.concat([train_data,test_data],axis=0,)
total_data=pd.get_dummies(data=total_data,drop_first=True)

train_data=total_data.iloc[:1460,:]

test_data=total_data.iloc[1460:,:]

XX=train_data.copy()

y=y_train.copy()
history=pd.DataFrame(columns=['itreation','will_deleted','pvalue','R_adjusted','rsquared'])

for x in range(len(XX.columns)):

        regressor_osl=sm.OLS(endog=y,exog=XX).fit()

        pvalues=regressor_osl.pvalues

        pvalues=pd.DataFrame(pvalues)

        p=pd.DataFrame(pvalues.reset_index())

        out=pd.DataFrame()

        out=p.loc[p[0]== max(p[0])]

        out=pd.Series.tolist(out)

        w1=out[0][0]

        R=regressor_osl.rsquared_adj

        r=regressor_osl.rsquared

        XX.pop(w1)

        history=history.append([{'itreation':x+1,'will_deleted':out[0][0],'pvalue':out[0][1],'R_adjusted':R,'rsquared':r}])
drop=history[history['pvalue']>.05]['will_deleted']

drop=list(drop)

drop
train_data.drop(columns=drop,inplace=True)

test_data.drop(columns=drop,inplace=True)
history[history['pvalue']<.05]
from sklearn.preprocessing import MinMaxScaler

sc=MinMaxScaler()

train_data=sc.fit_transform(train_data)

test_data=sc.transform(test_data)

train_data.shape[1]
model=keras.Sequential([keras.layers.Dense(500,input_shape=[train_data.shape[1]])

                 ,keras.layers.Dense(400,activation='relu')

                 ,keras.layers.Dense(300,activation='relu')

                 ,keras.layers.Dense(200,activation='relu')

                 ,keras.layers.Dense(100,activation='relu')

                ,keras.layers.Dense(1,activation='linear')

                 ])
model.compile(optimizer='adam',loss='mean_squared_error',metrics=['mean_squared_error'])

model.fit(train_data,y_train,batch_size=32,verbose=1,epochs=1000)



y_pre=model.predict(test_data)

test_id=test_id.astype(np.int32)
cols = ['Id', 'SalePrice']

submit_df = pd.DataFrame(np.hstack((test_id.reshape(-1,1),np.exp(y_pre.reshape(-1,1)))),columns=cols,dtype='Int32')
submit_df.to_csv('submission.csv', index=False)
submit_df
y_train.skew()