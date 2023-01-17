import numpy as np

import pandas as pd

import tensorflow as tf

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

from sklearn.model_selection import train_test_split

import keras

from sklearn import preprocessing

from sklearn.preprocessing import StandardScaler

from keras.models import Sequential

from keras.layers import Dense

train_ = pd.read_csv('../input/dark-side-of-moon-dataset/train.csv')

test_ = pd.read_csv('../input/dark-side-of-moon-dataset/test.csv')
train_.head(10)
test_.head(10)
print(train_.shape)

print(test_.shape)
train_['Lunation Number'].value_counts()
train_nans = train_.shape[0] - train_.dropna().shape[0]

test_nans = test_.shape[0] - test_.dropna().shape[0]

print(train_nans)

print(test_nans)
train_.isnull().sum()
cat = train_.select_dtypes(include= ['O'])

cat.apply(pd.Series.nunique)
t_data = [train_,test_]

for data in t_data:

    for x in data.columns:

        if data[x].dtype=='object':

            lbl = preprocessing.LabelEncoder()

            lbl.fit(list(data[x].values))

            data[x] = lbl.transform(list(data[x].values))



train_.head(10)
def normalize(dataset):

    dataNorm=((dataset-dataset.min())/(dataset.max()-dataset.min()))*20

    return dataNorm
train_norm= normalize(train_)

test_norm = normalize(test_)   
train_norm.head(10)
test_norm.head(10)
y = train_['Eclipse Duration (m)']
X = train_norm.drop(columns = ['Eclipse Duration (m)'])

X.shape
print(y.shape)
sc = StandardScaler()

X = sc.fit_transform(X)

test = sc.fit_transform(test_norm)
model = Sequential()

model.add(Dense(11,kernel_initializer='uniform',activation='relu',input_dim=11))

model.add(Dense(11,kernel_initializer='uniform',activation='relu'))

model.add(Dense(5,kernel_initializer='uniform',activation='relu'))

model.add(Dense(1,kernel_initializer='uniform',activation='linear'))

          
model.summary()
model.compile(optimizer='adam',loss ='mean_squared_error',metrics=['accuracy'])

model.fit(X,y,batch_size=32,nb_epoch=300)
test_norm.shape
test_norm.head(10)
test_norm = test_norm.drop(columns = ['ID'])
train_norm.head()
samp = pd.read_csv('../input/dark-side-of-moon-dataset/sample_submission.csv')

y_pred = model.predict(test_norm)
y_final = y_pred.astype(int).reshape(test_norm.shape[0])
samp['Eclipse Duration (m)'] = y_final

samp.to_csv('sample_ans.csv',index = False)