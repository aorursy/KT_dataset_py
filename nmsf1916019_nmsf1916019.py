#软件包配置

import sklearn

from sklearn import linear_model

from sklearn import preprocessing

from sklearn.model_selection import train_test_split

from sklearn.impute import SimpleImputer

from sklearn.metrics import mean_squared_error as MSE

from sklearn.metrics import median_absolute_error as MAE

from sklearn.ensemble import RandomForestRegressor



import os

import numpy as np

import pandas as pd

import lightgbm as lgb

import matplotlib.pyplot as plt

import seaborn as sns

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'



#读取文件数据 

traindatapath='../input/pubg-finish-placement-prediction/train_V2.csv'

testdatapath='../input/pubg-finish-placement-prediction/test_V2.csv'

traindata=pd.read_csv(traindatapath) #训练数据

testdata=pd.read_csv(testdatapath) #测试数据

#查看数据所有特征

traindata.info()

#查看训练集的前5条数据

traindata.head()

#traindata.trail()



#删除排名值为NaN的不合法数据

traindata.drop(2744604, inplace=True)



testId=testdata.loc[:,'Id']

# print(testId)



#选取特征数据

x=traindata[['walkDistance','boosts','weaponsAcquired','damageDealt','killPlace','kills','killStreaks','matchDuration','heals','rideDistance']]

y=traindata['winPlacePerc']

test=testdata[['walkDistance','boosts','weaponsAcquired','damageDealt','killPlace','kills','killStreaks','matchDuration','heals','rideDistance']]

#对空值数据的处理

imputer=SimpleImputer(missing_values=np.nan, strategy='mean')

x = imputer.fit_transform(x)

y = np.array(y).reshape(-1,1)

y = imputer.fit_transform(y)

#将数据映射到[-1,1]

scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1)).fit(x)

x = scaler.transform(x)

test = scaler.transform(test)

#print(test)

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.3)

print(x_train)

print(y_train)

print(x_test)

print(y_test)



#定义线性回归模型

#model=linear_model.LinearRegression()

#定义随机森林模型,n_estimators弱学习器的最大迭代次数,一般来说n_estimators太小，容易欠拟合，n_estimators太大，又容易过拟合

#max_features='sqrt'

#model = RandomForestRegressor(n_estimators=70, min_samples_leaf=3, max_features='sqrt',n_jobs=-1)

#定义轻量级boosting框架

model = lgb.LGBMRegressor(objective='mae',num_leaves=128,learning_rate=0.05,n_estimators=1000,metric='mae',verbose=1,random_state=42,bagging_fraction=0.7,feature_fraction=0.7)

model.fit(x_train,y_train)

y_predict = model.predict(x_test)

#print(y_predict)

predict_test = model.predict(test)

print(predict_test)

print(len(predict_test))

print(model.score(x_test, y_test))

 

 #合并Id和test_predict

testdata['winPlacePerc'] = predict_test

submission = testdata[['Id', 'winPlacePerc']]

submission.to_csv('submission.csv', index=False)

submission.head()
