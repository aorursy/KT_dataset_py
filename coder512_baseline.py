import os

print((os.listdir('../input/')))
import pandas as pd

import numpy as np

from sklearn import svm

import matplotlib.pyplot as plt

from sklearn import preprocessing

from sklearn.linear_model import LogisticRegression
df_train = pd.read_csv('../input/webclubrecruitment2019/TRAIN_DATA.csv')

df_test = pd.read_csv('../input/webclubrecruitment2019/TEST_DATA.csv')

test_index=df_test['Unnamed: 0'] #copying test index for later
df_train.head()
train_X = df_train.loc[:, 'V1':'V16']

train_y = df_train.loc[:, 'Class']

#norm_X=train_X[:]

#scaler = preprocessing.StandardScaler().fit(norm_X)

#scaled_X=scaler.transform(norm_X)    

#train_X[:]=scaled_X   

train_X=train_X.drop(['V5','V8','V9','V14','V15','V16'],axis=1)
model=svm.SVC(kernel='sigmoid',C=0.1)
model.fit(train_X,train_y)
df_test = df_test.loc[:, 'V1':'V16']

norm_X1=df_test[:]

scaler = preprocessing.StandardScaler().fit(norm_X1)

scaled_X1=scaler.transform(norm_X1)    

df_test[:]=scaled_X1 

df_test=df_test.drop(['V5','V8','V9','V14','V15','V16'],axis=1)

pred = model.predict(df_test)
result=pd.DataFrame()

result['Id'] = test_index

result['PredictedValue'] = pd.DataFrame(pred)

result
result.to_csv('output.csv', index=False)