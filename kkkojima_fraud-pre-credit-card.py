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
#再現性の確保

import os

import numpy as np

import random as rn

import tensorflow as tf



os.environ['PYTHONHASHSEED'] = '0'

np.random.seed(7)

rn.seed(7)
train = pd.read_csv('/kaggle/input/1056lab-fraud-detection-in-credit-card/train.csv', index_col=0)

test = pd.read_csv('/kaggle/input/1056lab-fraud-detection-in-credit-card/test.csv', index_col=0)
train.head()
import collections

a = collections.Counter(train['Class'])

print(a)
print(train.isnull().sum())
print(test.isnull().sum())
import seaborn as sns

import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(15, 9))

sns.heatmap(train.corr(), cmap='BuPu', annot=True,fmt='.2f')
#X_train = train.drop('Class', axis=1).values

#X_train = train[['V1', 'V2', 'V3', 'V4', 'V5', 'V7', 'V9', 'V10', 'V11', 'V12', 'V14', 'V16', 'V17']].values

X_train = train[['V3', 'V7', 'V10', 'V12', 'V14', 'V16', 'V17']].values #←相関の値が0.2以上のもの

y_train = train[['Class']]

from sklearn.model_selection import train_test_split

X_learn, X_valid, y_learn, y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=0)

print('↓学習用データ')

print(X_learn.shape, X_valid.shape)

print('↓検証用データ')

print(y_learn.shape, y_valid.shape)
#不均衡データ対策

from imblearn.over_sampling import SMOTE

ros = SMOTE(random_state=0)#, ratio={1:100000})

X_res, y_res = ros.fit_sample(X_learn, y_learn)

print(X_res.shape, y_res.shape)
a = collections.Counter(y_res)

print(a)
from keras.utils.np_utils import to_categorical

Y_learn = to_categorical(y_learn)

Y_res = to_categorical(y_res)

Y_valid = to_categorical(y_valid)
#学習モデルの作成

import tensorflow as tf

from keras.models import Sequential

from keras.layers import Dense, Activation, Dropout, BatchNormalization

from keras import optimizers

from tensorflow.keras.metrics import AUC

AUC = AUC()



model = Sequential()

model.add(Dense(128, activation='relu', input_dim=X_res.shape[1]))

model.add(Dense(64, activation='relu'))

model.add(Dropout(0.25))

model.add(Dense(32, activation='relu'))

model.add(Dropout(0.25))

model.add(Dense(2, activation='softmax'))

model.compile(optimizer=optimizers.Adam(lr=2.5e-4),

              loss='binary_crossentropy',

              metrics=['accuracy', AUC])
model.fit(X_res, Y_res, epochs=1, batch_size=32)
model.evaluate(X_valid, Y_valid, batch_size=32)
#学習データに対する精度（AUC）

from sklearn.metrics import roc_curve, auc

predictions = model.predict(X_learn)

precision, recall, thresholds = roc_curve(Y_learn[:,1], predictions[:,1])

score = auc(precision, recall)

score

print('AUC(1)',score)
#検証用データに対する精度（AUC）

predictions = model.predict(X_valid)

precision, recall, thresholds = roc_curve(Y_valid[:,1], predictions[:,1])

score = auc(precision, recall)

print('AUC(1)',score)
X, y = ros.fit_sample(X_train, y_train)

Y = to_categorical(y)

print(X.shape, Y.shape)
#学習モデルの作成

import tensorflow as tf

from keras.models import Sequential

from keras.layers import Dense, Activation, Dropout, BatchNormalization

from keras import optimizers

from tensorflow.keras.metrics import AUC

AUC = AUC()



model = Sequential()

model.add(Dense(128, activation='relu', input_dim=X.shape[1]))

model.add(Dense(64, activation='relu'))

model.add(Dropout(0.25))

model.add(Dense(32, activation='relu'))

model.add(Dropout(0.25))

model.add(Dense(2, activation='softmax'))

model.compile(optimizer=optimizers.Adam(lr=2.5e-4),

              loss='binary_crossentropy',

              metrics=['accuracy', AUC])
model.fit(X, Y, epochs=1, batch_size=32)
#test1 = test[['V1', 'V2', 'V3', 'V4', 'V5', 'V7', 'V9', 'V10', 'V11', 'V12', 'V14', 'V16', 'V17']].values

test1 = test[['V3', 'V7', 'V10', 'V12', 'V14', 'V16', 'V17']].values 

pre = model.predict(test1)
p = pd.DataFrame(pre)

p.head(30)
submit = pd.read_csv('/kaggle/input/1056lab-fraud-detection-in-credit-card/sampleSubmission.csv')

submit['Class'] = pre[:,1]

submit.to_csv('pre.csv', index=False)