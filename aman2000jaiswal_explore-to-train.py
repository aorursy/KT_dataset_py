# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np

import pandas as pd
df = pd.read_csv('../input/lish-moa/train_features.csv')
df.head()
df_nonscored = pd.read_csv('../input/lish-moa/train_targets_nonscored.csv')
df_nonscored.head()
df_scored = pd.read_csv('../input/lish-moa/train_targets_scored.csv')
df_scored.head()
df_sample = pd.read_csv('../input/lish-moa/sample_submission.csv')
df_sample.head()
df_test = pd.read_csv('../input/lish-moa/test_features.csv')
df_test.head()
print(df.shape)

print(df_sample.shape)

print(df_scored.shape)

print(df_nonscored.shape)

print(df_test.shape)
df_cols = df.columns

df_test_cols = df_test.columns

df_sample_cols = df_sample.columns

df_scored_cols = df_scored.columns

df_nonscored_cols = df_nonscored.columns
common = set(df_scored_cols).intersection(set(df_sample_cols))

len(common)
common = set(df_nonscored_cols).intersection(set(df_sample_cols))

len(common)
train = df.drop(['sig_id','cp_type'],axis=1)

test = df_test.drop(['sig_id','cp_type'],axis=1)

target = df_scored.drop(['sig_id'],axis=1)
print(any(train.isnull().any()))

print(any(target.isnull().any()))

print(any(target.isnull().any()))
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

train['cp_dose'] = le.fit_transform(train.cp_dose)

test['cp_dose'] = le.transform(test.cp_dose)
from sklearn.preprocessing import MinMaxScaler

mms = MinMaxScaler()

mms.fit(train)
train = mms.transform(train)
test = mms.transform(test)
train.shape
target.shape
test.shape
import tensorflow as tf

import keras 

from keras.models import Sequential

from keras.layers import Dense,Dropout,BatchNormalization

from tensorflow.keras import initializers
keras.backend.clear_session()

model = Sequential()

model.add(Dense(3000,activation='relu',input_shape= (874,),kernel_initializer = keras.initializers.GlorotNormal()))

# model.add(Dropout(0.4))

# model.add(BatchNormalization())

# model.add(Dense(2000,activation='relu',kernel_initializer = keras.initializers.GlorotNormal()))

model.add(Dense(2000,activation='relu',kernel_initializer = keras.initializers.GlorotUniform()))

# model.add(Dropout(0.4))

# model.add(BatchNormalization())

# model.add(Dense(1000,activation='relu',kernel_initializer = keras.initializers.GlorotNormal()))

# model.add(Dropout(0.2))

# model.add(BatchNormalization())

# model.add(Dense(1000,activation='relu',kernel_initializer = keras.initializers.GlorotNormal()))

# model.add(Dropout(0.2))

# model.add(BatchNormalization())

# model.add(Dense(500,activation='relu',kernel_initializer = keras.initializers.GlorotNormal()))

# model.add(Dropout(0.2))

model.add(Dense(206,activation='softmax',kernel_initializer = keras.initializers.RandomUniform(minval=0., maxval=1.)))

model.compile('adam','categorical_crossentropy',metrics = ['accuracy'])

model.summary()
hist = model.fit(train,target,batch_size = 64, validation_split=0.2,epochs=1)
from sklearn.linear_model import LinearRegression

lr = LinearRegression()

lr.fit(train,target)
import matplotlib.pyplot as plt
plt.figure(figsize=(10,7))

plt.subplot(1,2,1)

plt.plot(hist.history['accuracy'],label='accuracy')

plt.plot(hist.history['loss'],label='loss')

plt.legend()

plt.title("training set")

plt.grid()

plt.subplot(1,2,2)

plt.plot(hist.history['val_accuracy'],label='val_accuracy')

plt.plot(hist.history['val_loss'],label='val_loss')

plt.legend()

plt.title("validation set")

plt.grid()

plt.ylim((0,4))
lrpred = lr.predict(test)
pred = model.predict(test)
pred[1:3]
df_sample.iloc[:,1:] = pred+lrpred
df_sample.shape
df_sample.to_csv('submission.csv',index=False)