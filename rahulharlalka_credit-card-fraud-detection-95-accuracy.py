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
dataset=pd.read_csv('/kaggle/input/creditcardfraud/creditcard.csv')

dataset.head()
print("null values in the dataset= ",dataset.isnull().sum().sum())
dataset.info()
import seaborn as sns

import matplotlib.pyplot as plt

sns.countplot(dataset.Class)

plt.savefig("class distributions")



print(dataset['Class'].value_counts())
fraud_data=dataset[dataset['Class']==1]

non_fraud_data=dataset[dataset['Class']==0]

non_fraud_data=non_fraud_data.sample(n=750,random_state=18)

#shuffling the data randomly for better data training

from sklearn.utils import shuffle 

data=pd.concat([fraud_data,non_fraud_data],ignore_index=True)

data=shuffle(data)

sns.countplot(data.Class)

plt.savefig("final class distribution")
data
from sklearn.preprocessing import StandardScaler

x_data=data.drop(labels="Class",axis=1)

y_data=data.Class



scaler=StandardScaler()

scaler.fit(x_data)

x_data=scaler.transform(x_data)

from sklearn.model_selection import train_test_split as tts

x_train,x_test,y_train,y_test=tts(x_data,y_data,test_size=0.3,random_state=18)

x_train=x_train.reshape(len(x_train),30,1)

x_test=x_test.reshape(len(x_test),30,1)

print("x_train data shape=",x_train.shape)

print("x_test data shape=",x_test.shape)

print("y_train data shape=",y_train.shape)

print("y_test data shape=",y_test.shape)

import tensorflow as tf

from tensorflow import keras

from tensorflow.keras import models,layers
model=models.Sequential()

model.add(layers.Conv1D(64,7,activation='relu',input_shape=(30,1)))

model.add(layers.Conv1D(32,5,activation='relu',kernel_regularizer='l2'))

model.add(layers.Dropout(0.1))

model.add(layers.Conv1D(16,3,activation='relu',kernel_regularizer='l2'))

model.add(layers.Flatten())

model.add(layers.Dense(256,activation='relu',kernel_regularizer='l2'))

model.add(layers.Dense(128,activation='relu',kernel_regularizer='l2'))

model.add(layers.Dropout(0.1))

model.add(layers.Dense(2,activation='softmax',kernel_regularizer='l2'))

model.summary()
keras.utils.plot_model(model)
weights={0:0.3,1:0.7}

model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

model.fit(x_train,y_train,epochs=20,validation_split=0.2)
from sklearn.metrics import accuracy_score, confusion_matrix

y_predict=model.predict(x_test)

y=[]

for i in y_predict:

    y.append(np.argmax(i))



print(accuracy_score(y_test,y))

confusion_matrix(y_test,y)