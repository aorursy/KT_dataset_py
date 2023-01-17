# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train.csv')

valid  = pd.read_csv('../input/test.csv')
print("Shape of Train Data : ", train.shape)

print("Shape of Validation Data : ",  valid.shape)
print("Data Type in the Training Data : ")

print(train.dtypes)

print(valid.dtypes)
print('Summary Statistics:')

for col in train.columns:

    print('Distinct Values, min and, max of : ',col, train[col].nunique(), min(train[col]), max(train[col]))
print('Distribution of Events : ')

dfOut = train['target'].value_counts().reset_index()

dfOut.columns = ['target', 'event']

dfOut['eventRate'] = dfOut['event']/sum(dfOut['event'])

print(dfOut)
print(' --- Risk Table (Different Approach) --- ')

for col in train.columns:

    if(col not in ['id','target','ps_reg_03','ps_car_12','ps_car_13','ps_car_14','ps_car_15']):

        print("Feature : ", col)

        dfOut = train.groupby(col)['target'].agg({'sum' : 'sum', 'count' : 'count'}).reset_index()

        dfOut['eventOdd'] = dfOut['sum']/dfOut['count'] * 100

        dfOut[col + '_eventRate']= dfOut['sum']/sum(dfOut['sum']) * 100

        dfOut.sort_values(col + '_eventRate', ascending = False, inplace = True)

        dfOut.drop(['sum','count','eventOdd'], axis = 1, inplace = True)

        train = pd.merge(train, dfOut, on = col, how = 'inner')

        train.drop(col, axis = 1, inplace = True)



pd.set_option('display.max_columns',None)       

print(train.head())

print('--- XGBoost ---')

import random

from xgboost import XGBRegressor



train['randomNumber'] = [random.uniform(0,1) for x in range(train.shape[0])]



dfTrain = train.query('randomNumber<=.7')

dfTest  = train.query('randomNumber>.7')



print('--- Distribuion of Event in Train and Test Datasets')

print('--- Train ---\n', dfTrain['target'].value_counts())

print('--- Test ---\n', dfTest['target'].value_counts())



colsToKeep = [x for x in train.columns if x not in ('id','target','randomNumber')]

xTrain = dfTrain[colsToKeep].apply(lambda x: x).values

yTrain = dfTrain['target'].values

xTest  = dfTest[colsToKeep].apply(lambda x: x).values

yTest  = dfTest['target'].values



reg = XGBRegressor()

reg.fit(xTrain, yTrain)

yPred = reg.predict(xTest)



from sklearn.metrics import log_loss, accuracy_score

print('Log Loss:\n', log_loss(yTest, yPred))

print('--- Model Validation ---')

xValid = valid.drop(['id'], axis = 1).values



yOut = reg.predict(xValid)



outDf = pd.DataFrame()

outDf['id'] = valid['id']

outDf['target'] = yOut



outDf.to_csv('100_test.csv', index = False)