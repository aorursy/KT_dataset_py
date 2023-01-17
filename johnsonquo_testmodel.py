import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
##from sklearn import datasets, linear_model,cross_validation
from sklearn.metrics import precision_score, recall_score, accuracy_score,log_loss
import xgboost as xgb
os.listdir('../input')
nav=pd.read_csv('../input/fundNav.csv',encoding='big5')
indexdata = pd.read_csv('../input/index.csv')
info=pd.read_csv('../input/fundInfo.csv',encoding='big5')
weekdata = pd.read_csv('../input/fundWeekData.csv',encoding = 'big5')
trainData = pd.read_csv('../input/fundData.csv',encoding = 'big5')
predictData = pd.read_csv('../input/testFund.csv',encoding = 'big5')
predictData['code']=predictData['Unnamed: 0']
predictData['ret']=predictData['weekly_ret(%)']
predictData = predictData[['code','ret']]
predictData 

codeList = np.unique(trainData['code'])
len(codeList)
trainData
def accuracy(predict_y, test_y):
    loss = 0
    for i in range(len(predict_y)):
        loss += (predict_y[i]-test_y[i])**2
        ##loss += abs(predict_y[i]-test_y[i])
    return loss
trainData1 = pd.DataFrame({'week':[i for i in range(200)] })
trainData1[codeList[1]] = 1
trainData1
for index in codeList:
    trainData1[index] = pd.Series(trainData[(trainData['code']==index) & (trainData['trueWeek'] >= 1) & (trainData['trueWeek']<=200)]['weekRet'].values.tolist())
train_Data = trainData1.iloc[0:198]
test_Data = trainData1.iloc[1:199]
train_Data
aIndex = indexdata.query('代碼 == "OC72   "')
dataSet = indexdata.pivot(index='年月',columns='簡稱',values='指數')
##dataSet = pd.merge(fund,dataSet, left_on ='date', right_on='年月').dropna()
pre_cor_data = weekdata
pre_cor_data = weekdata.drop(['code','week','date','open'],axis= 1)
pre_cor_data['weekRet'] = pre_cor_data['weekRet'].shift(-1)
pre_cor_data = pre_cor_data.dropna()
##cor_matrix = pre_cor_data.corr()['ret']##.sort_values('原幣值')


train_x = train_Data.drop(['week'],axis = 1)
train_y = test_Data.drop(['week'],axis = 1)
predict_x = pd.DataFrame(predictData['weekRet'].values).transpose()
predict_x.columns = codeList
loss = accuracy(np.zeros(len(test_y)),test_y)
loss
regressionModel = linear_model.LinearRegression()
regressionModel.fit(train_x,train_y)
predict_y = regressionModel.predict(predict_x)
predict_y
loss = accuracy(predict_y,test_y)
loss
train_x = train_x.ix[:,codeList]
predict_x


train_x
for index in codeList:
    xgbm = xgb.XGBRegressor().fit(train_x, train_y[index])
    predict_y = xgbm.predict(predict_x)
    print("fund:",index,"ans:",predict_y)
loss = accuracy(predict_y,test_y)
loss
dnn = Sequential()
dnn.add(Dense(output_dim = 3, activation = 'sigmoid'))
# Adding the second hidden layer
dnn.add(Dense(output_dim = 3, activation = 'sigmoid'))
dnn.add(Dense(output_dim = 3, activation = 'sigmoid'))
# Adding the output layer
dnn.add(Dense(output_dim = 1, activation = 'sigmoid'))
dnn.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy',"mae"])
train_x = np.array(train_x)
train_y = np.array(train_y)
test_x = np.array(test_x)
dnn.fit(x=train_x, y=train_y,epochs=6)
predict_y = dnn.predict(test_x)
predict_y = np.squeeze(list(predict_y))
predict_y
loss = accuracy(predict_y,test_y)
loss