# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/breast-cancer-wisconsin-data/data.csv")
df.info()

# checking for null values
df.loc[df['diagnosis']=='M', 'benign_0__mal_1'] = 1

df.loc[df['diagnosis']=='B', 'benign_0__mal_1'] = 0

df.drop(columns='diagnosis',inplace=True)
df = df.drop(columns = ['id','Unnamed: 32'])
df.describe().transpose()

# check the statistical distributions of features and labels
sns.countplot(x='benign_0__mal_1', data=df)
df.corr()['benign_0__mal_1'].sort_values()
plt.figure(figsize=(12,8))

df.corr()['benign_0__mal_1'].sort_values().plot(kind='bar')
plt.figure(figsize=(12,12))

sns.heatmap(df.corr())
X = df.drop('benign_0__mal_1', axis=1).values

y = df['benign_0__mal_1'].values
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25,random_state=101)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
import tensorflow as tf

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Activation,Dropout
model = Sequential()



model.add(Dense(30,activation='relu'))

model.add(Dense(15,activation='relu'))

# for binary classification

model.add(Dense(1,activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam')
model.fit(x=X_train, y=y_train, epochs=600, validation_data=(X_test,y_test))
model_loss = pd.DataFrame(model.history.history)
model_loss.plot()
model = Sequential()

model.add(Dense(units=30,activation='relu'))

model.add(Dense(units=15,activation='relu'))

model.add(Dense(units=1,activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam')
from tensorflow.keras.callbacks import EarlyStopping
early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=25)
model.fit(x=X_train, 

          y=y_train, 

          epochs=600,

          validation_data=(X_test, y_test), verbose=1,

          callbacks=[early_stop]

          )
model_loss = pd.DataFrame(model.history.history)

model_loss.plot()
from tensorflow.keras.layers import Dropout
model = Sequential()



model.add(Dense(30,activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(15,activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(1,activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam')
model.fit(x=X_train, 

          y=y_train, 

          epochs=600,

          validation_data=(X_test, y_test), verbose=1,

          callbacks=[early_stop]

          )
model_loss = pd.DataFrame(model.history.history)
model_loss.plot()
predictions = model.predict_classes(X_test)
from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(y_test,predictions))
print(confusion_matrix(y_test,predictions))