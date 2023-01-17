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
import os

import time

import datetime

import numpy as np

import pandas as pd



# Keras

import tensorflow as tf

import tensorflow.keras as keras

from tensorflow.keras import layers

from tensorflow.keras import models

from tensorflow.keras import callbacks

from tensorflow.keras import backend as K



# Standard ML stuff

from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import MinMaxScaler, StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report

from sklearn.metrics import accuracy_score

from sklearn.manifold import TSNE

from sklearn.decomposition import PCA, TruncatedSVD, FastICA

from sklearn.random_projection import GaussianRandomProjection, SparseRandomProjection



# Oversampling of minority class 'Churn customers'

from imblearn.over_sampling import SMOTE



# Plotting

import matplotlib.pyplot as plt
df=pd.read_csv('/kaggle/input/telco-customer-churn/WA_Fn-UseC_-Telco-Customer-Churn.csv')

df.head()
print('features: \n ', df.columns.tolist())
df.isnull().sum()
# data preprocessing

print(df.gender.value_counts())

df['Female']=df['gender']=='Female'

df['Female']=df['Female'].astype(int)

df.drop('gender', axis=1, inplace=True)
print(df.Female.value_counts())

# Male:0, Female:1
print(df.Partner.value_counts())

df['Partner']=df['Partner'].map({'Yes':1, 'No':0})

print(df.Partner.value_counts())
print(df.Dependents.value_counts())

df['Dependents'] = df['Dependents'].map({'Yes': 1, 'No': 0})

print(df.Dependents.value_counts())
print(df.PhoneService.value_counts())

df['PhoneService'] = df['PhoneService'].map({'Yes': 1, 'No': 0})

print(df.PhoneService.value_counts())
print(df.MultipleLines.value_counts())

df['MultipleLines'] = df['MultipleLines'].map({'Yes' : 1, 'No' : 0, 'No phone service' : 0})

print(df.MultipleLines.value_counts())
df.InternetService.value_counts()
df['OnlineSecurity'] = df['OnlineSecurity'].map({'Yes' : 1, 'No' : 0, 'No internet service' : 0})

print(df.OnlineSecurity.value_counts())
df['OnlineBackup'] = df['OnlineBackup'].map({'Yes' : 1, 'No' : 0, 'No internet service' : 0})

print(df.OnlineBackup.value_counts())
df['DeviceProtection'] = df['DeviceProtection'].map({'Yes' : 1, 'No' : 0, 'No internet service' : 0})

print(df.DeviceProtection.value_counts())
df['TechSupport'] = df['TechSupport'].map({'Yes' : 1, 'No' : 0, 'No internet service' : 0})

print(df.TechSupport.value_counts())
df['StreamingTV'] = df['StreamingTV'].map({'Yes' : 1, 'No' : 0, 'No internet service' : 0})

print(df.StreamingTV.value_counts())
df['StreamingMovies'] = df['StreamingMovies'].map({'Yes' : 1, 'No' : 0, 'No internet service' : 0})

print(df.StreamingMovies.value_counts())

df['PaperlessBilling'] = df['PaperlessBilling'].map({'Yes': 1, 'No': 0})

print(df.PaperlessBilling.value_counts())
df.Contract.value_counts()
df.PaymentMethod.value_counts()
df['Churn']=df['Churn'].map({'Yes':1, 'No':0})

print(df.Churn.value_counts())
df.head()
# remove empty data but not null 

df['TotalCharges']=df['TotalCharges'].replace(" ", np.nan)

print("Missing Values in TotalCharges: ", df['TotalCharges'].isnull().sum())



df=df[df["TotalCharges"].notnull()]

df=df.reset_index()[df.columns]

print("Missing Values in TotalCharges: ", df['TotalCharges'].isnull().sum())



df["TotalCharges"] = df["TotalCharges"].astype(float)

print("dType TotalCharges: ", df['TotalCharges'].dtype)
numeric_cols=['MonthlyCharges', 'TotalCharges', 'tenure']

target_col=['Chrun']

ignored_cols=['customerID']

categorical_cols=['SeniorCitizen', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod', 'Female']

categorical_cols=[col for col in categorical_cols if col not in target_col + ignored_cols]

for col in categorical_cols:

    df[col] = LabelEncoder().fit_transform(df[col])
len(categorical_cols)
df
df[numeric_cols]
df
train_df, test_df=train_test_split(df, test_size=0.15, random_state=42)

print(train_df.shape)
x=train_df[numeric_cols+categorical_cols]

y=train_df['Churn']
x_train, x_val, y_train, y_val=train_test_split(x,y,test_size=0.2, random_state=42)
#x_train=x_train.values

#y_train=y_train.values
x_train

y_train
#x_val=x_val.values

#y_val=y_val.values
x_test=test_df[numeric_cols+categorical_cols]

y_test=test_df['Churn']
# 전처리,정규화

from sklearn.preprocessing import StandardScaler

scaler=StandardScaler()

x_train=scaler.fit_transform(x_train)

x_test=scaler.transform(x_test)







from tensorflow.keras.layers.experimental.preprocessing import Normalization

normalizer=Normalization(axis=-1)

normalizer.adapt(x_train)

x_train=normalizer(x_train).numpy()

x_test=normalizer(x_test)

model=tf.keras.Sequential()

model.add(tf.keras.layers.Dense(10,input_shape=(19

                                               ,), activation='relu'))

model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy',metrics=['accuracy'])

model.fit(x_train, y_train, epochs=100, batch_size=20, validation_split=0.2)
model.summary()
predictions=model.predict(x_test)

print(predictions, y_test)
loss, accuracy = model.evaluate(x_val,y_val)

print("Accuracy :%.2f"% (accuracy*100))

print("Loss:%.2f"%(loss*100))
model=tf.keras.Sequential()

model.add(tf.keras.layers.Dense(8,input_shape=(19

                                               ,), activation='sigmoid'))

model.add(tf.keras.layers.Dense(1, activation='softmax'))

model.compile(optimizer='adam', loss='binary_crossentropy',metrics=['accuracy'])
model.summary()
model.fit(x_train, y_train, epochs=100, batch_size=16, validation_split=0.2)
predictions=model.predict(x_test)

print(predictions, y_test)
loss, accuracy = model.evaluate(x_val,y_val)

print("Accuracy :%.2f"% (accuracy*100))

print("Loss:%.2f"%(loss*100))
inputs = tf.keras.Input(shape=(19,))

x = tf.keras.layers.Dense(8, activation=tf.nn.relu)(inputs)

outputs = tf.keras.layers.Dense(1, activation=tf.nn.softmax)(x)

model = tf.keras.Model(inputs=inputs, outputs=outputs)

model.compile(optimizer='adam', loss='binary_crossentropy',metrics=['accuracy'])
model.summary()
model.fit(x_train, y_train, epochs=100, batch_size=19, validation_split=0.2)
loss, accuracy = model.evaluate(x_val,y_val)

print("Accuracy :%.2f"% (accuracy*100))

print("Loss:%.2f"%(loss*100))
model = keras.Sequential([

layers.Input(shape=19),

layers.Dense(8, activation='relu'),

layers.Dense(16, activation='sigmoid'),

layers.Dense(1, activation='softmax')])

model.compile(optimizer='adam', loss='binary_crossentropy',metrics=['accuracy'])
model.fit(x_train, y_train, epochs=100, batch_size=19, validation_split=0.2)
predictions=model.predict(x_test)

print(predictions, y_test)
loss, accuracy = model.evaluate(x_val,y_val)

print("Accuracy :%.2f"% (accuracy*100))
from sklearn.metrics.classification import classification_report

from sklearn.metrics import confusion_matrix



print(classification_report(y_test, predictions))
hist=model.fit(x_train, y_train, epochs=100, batch_size=256,validation_split=0.2)



import matplotlib.pyplot as plt

plt.plot(hist.history['loss'])