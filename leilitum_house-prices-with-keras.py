# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib



import matplotlib.pyplot as plt

from scipy.stats import skew

from scipy.stats.stats import pearsonr



# Input data files are available in the "../input/" directory.

%matplotlib inline



train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")
all_data = pd.concat((train.loc[:,'MSSubClass':'SaleCondition'],test.loc[:,'MSSubClass':'SaleCondition']))
train['SalePrice'] = np.log1p(train['SalePrice'])



numeric_features = all_data.dtypes[all_data.dtypes != 'object'].index

skewed_features = train[numeric_features].apply(lambda x: skew(x.dropna()))

skewed_features= skewed_features[skewed_features>0.75]

skewed_features = skewed_features.index



all_data[skewed_features] = np.log1p(all_data[skewed_features])
total = all_data.isnull().sum().sort_values(ascending=False)

percent = (all_data.isnull().sum()/all_data.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_data.head(10)
all_data = all_data.drop(missing_data[missing_data['Total']>100].index,1)

all_data = all_data.fillna(all_data.mean())
all_data = pd.get_dummies(all_data)
X_train = all_data[:train.shape[0]]

X_test = all_data[train.shape[0]:]

y = train.SalePrice
from keras.layers import Dense, Activation

from keras.models import Sequential

from keras.regularizers import l1

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split
from keras import regularizers
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
X_tr, X_val, y_tr, y_val = train_test_split(X_train, y, random_state = 3)

X_tr.shape
model = Sequential()

model.add(Dense(64,input_shape=[X_tr.shape[1],],kernel_regularizer=regularizers.l2(0.01)))

model.add(Activation('relu'))

model.add(Dense(1))

model.add(Activation('relu'))

model.compile(loss = "mse", optimizer = "adam")

model.fit(X_tr,y_tr,validation_data=(X_val, y_val))

score = model.evaluate(X_val, y_val)
predictions=model.predict(X_test)
predictions
submission_col = 'SalePrice'

submission_target = 'test_sub1.csv'

submission_name = '../input/sample_submission.csv'

submission = pd.read_csv(submission_name)



submission[submission_col] = np.exp(predictions)

submission.to_csv(submission_target,index=False)