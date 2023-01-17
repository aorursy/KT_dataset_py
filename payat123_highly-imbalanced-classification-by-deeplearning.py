# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
seed = 7
np.random.seed(seed)

import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix, classification_report

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
df = pd.read_csv('../input/creditcard.csv')
df.head()
# make dataset more balance by randomly removing majority class
df_major = df[df['Class']==0]
df_minor = df[df['Class']==1]
major_count, minor_count = df.Class.value_counts()
print(major_count)
print(minor_count)
print('ratio imbalance dataset:',major_count/minor_count)
# shuffle both major and minor classes
df_major = shuffle(df_major, random_state=42)
df_minor = shuffle(df_minor, random_state=42)

# split minor class into train 80% and dev 20%
perc = 0.8
minor_data_train = int(perc*minor_count)
df_minor_train = df_minor[0:minor_data_train]
df_minor_dev = df_minor[minor_data_train:]
# rebalance training set into the ratio
ratio_imb = 2.0
major_data_train = int(ratio_imb*minor_data_train)
df_major_train = df_major[0:major_data_train]
df_major_dev = df_major[major_data_train:int(ratio_imb*major_data_train)]
df_major_test = df_major[int(ratio_imb*major_data_train):]
major_c = df_major_train.Class.value_counts()
minor_c = df_minor_train.Class.value_counts()
print('ratio imbalance dataset:',int(major_c)/int(minor_c))
# concat to make df_train, df_dev and df_test
df_train = pd.concat([df_major_train, df_minor_train], axis=0)
df_dev = pd.concat([df_major_dev, df_minor_dev], axis=0)
df_test = pd.concat([df_major_test, df_minor_dev], axis=0)
# shuffle agian make sure they are not orderical
df_train = shuffle(df_train, random_state=42)
df_dev = shuffle(df_dev, random_state=42)
df_test = shuffle(df_test, random_state=42)
feature_train = df_train.drop(['Time', 'Amount'], axis=1)
target_train = df_train['Class']
feature_dev = df_dev.drop(['Time', 'Amount'], axis=1)
target_dev = df_dev['Class']
scalar = StandardScaler()
scalar.fit(feature_train)
X_train = scalar.transform(feature_train)
y_train = target_train
X_dev = scalar.transform(feature_dev)
y_dev = target_dev
X_train.shape
y_train.shape
# create model three layers
model = Sequential()
model.add(Dense(100, input_dim=29, kernel_initializer='uniform',  activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(100, kernel_initializer='uniform',  activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1,kernel_initializer='uniform', activation='sigmoid'))

# compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# fit model
train_history = model.fit(X_train, y_train, epochs=50, batch_size=128, verbose=0)
plt.plot(train_history.history['acc'])
X_dev.shape
y_dev.shape
prediction = model.predict_classes(X_dev)
print(classification_report(y_dev, prediction))
print(confusion_matrix(y_dev, prediction))
feature_test = df_test.drop(['Time', 'Amount'], axis=1)
target_test = df_test['Class']
X_test = scalar.transform(feature_test)
y_test = target_test
X_test.shape
y_test.shape
prediction_test = model.predict_classes(X_test)
print(classification_report(y_test, prediction_test))
print(confusion_matrix(y_test, prediction_test))
