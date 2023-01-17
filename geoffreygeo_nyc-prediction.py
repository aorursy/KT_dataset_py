# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import dask.dataframe as dd
import seaborn as sns
import matplotlib.pyplot as plt
from __future__ import absolute_import, division, print_function

import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LinearRegression
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
%%time
#path of the file 
file = '../input/train.csv'
file_train='../input/test.csv'
# Set columns to most suitable type to optimize for memory usage
traintypes = {'fare_amount': 'float32',
              'pickup_datetime': 'str', 
              'pickup_longitude': 'float32',
              'pickup_latitude': 'float32',
              'dropoff_longitude': 'float32',
              'dropoff_latitude': 'float32',
              'passenger_count': 'uint8'}

testtypes = { 'pickup_datetime': 'str', 
              'pickup_longitude': 'float32',
              'pickup_latitude': 'float32',
              'dropoff_longitude': 'float32',
              'dropoff_latitude': 'float32',
              'passenger_count': 'uint8'}

cols = list(traintypes.keys())
cols_train=list(testtypes.keys())
df_train = pd.read_csv(file,usecols=cols,dtype=traintypes,nrows = 500)
df_test=pd.read_csv(file_train,usecols=cols_train,dtype=testtypes,nrows=500)
%%time
display(df_train.describe())
display(df_test.describe())
df_test.head()
def distance_between(df):
    lat1=df['pickup_latitude']
    lat2=df['dropoff_latitude']
    lon1=df['pickup_longitude']
    lon2=df['dropoff_longitude']
    dist = np.degrees(np.arccos(np.minimum(1,np.sin(np.radians(lat1)) * np.sin(np.radians(lat2)) + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.cos(np.radians(lon2 - lon1))))) * 60 * 1.515 * 1.609344
    return dist
def split_date_time(df):
    return df['pickup_datetime'].dt.year,df['pickup_datetime'].dt.month,df['pickup_datetime'].dt.day,df['pickup_datetime'].dt.dayofweek,df['pickup_datetime'].dt.hour

df_train[['date','time','time-zone']] =df_train['pickup_datetime'].str.split(' ', expand=True)
df_train[['year','month','date']] =df_train['date'].str.split('-',expand=True)
df_test[['date','time','time-zone']] =df_test['pickup_datetime'].str.split(' ', expand=True)
df_test[['year','month','date']] =df_test['date'].str.split('-',expand=True)
df_test[['HH','MM','SS']]=df_test['time'].str.split(':',expand=True)
df_train[['HH','MM','SS']]=df_test['time'].str.split(':',expand=True)
df_train['total_distance']=df_train.apply(distance_between,axis=1)
df_test['total_distance']=df_test.apply(distance_between,axis=1)
df_train.shape
df_train= df_train.drop(df_train[(df_train['total_distance']==0)].index, axis = 0)
df_train.head()
df_train.head()
def plot_history(history):
  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Mean Abs Error [1000$]')
  plt.plot(history.epoch, np.array(history.history['mean_absolute_error']),
           label='Train Loss')
  plt.plot(history.epoch, np.array(history.history['val_mean_absolute_error']),
           label = 'Val loss')
  plt.legend()
  plt.ylim([0, 5])
#ploting the fareamount again the number of passengers
sns.boxplot(x='passenger_count',y='fare_amount', data= df_train)
sns.regplot(x='total_distance',y='fare_amount',fit_reg=False, ci=None, truncate=True,data=df_train)
sns.boxplot(x=df_train['passenger_count'])
sns.swarmplot(x=df_train['fare_amount'])
df_train_label=df_train['fare_amount']
df_train_data=df_train.drop(['fare_amount','pickup_datetime','time-zone','time','MM','SS'],axis=1)
df_test_data=df_test.drop(['pickup_datetime','time-zone','time','MM','SS'],axis=1)
print("{}\n{}".format(df_train_data.shape,df_test_data.shape))
df_train_data.head()
epochs=30
def build_model():
  model = keras.Sequential([
    keras.layers.Dense(64, activation=tf.nn.relu,
                       input_shape=(df_train_data.shape[1],)),
    keras.layers.Dense(64, activation=tf.nn.relu),
    keras.layers.Dense(1)
  ])

  optimizer = tf.train.RMSPropOptimizer(0.001)

  model.compile(loss='mse',
                optimizer=optimizer,
                metrics=['mae'])
  return model

model = build_model()
model.summary()
#create model and train the model 
trained_model=model.fit(df_train_data,df_train_label,epochs=epochs,validation_split=0.2)
test_predictions=model.predict(df_test_data)
print("actual: "+str(df_train_label[0:5].values))
print("pred:   "+str(test_predictions[24:35]))
clf=LinearRegression()
clf.fit(df_train_data,df_train_label)
predicted_values=clf.predict(df_test_data)
print("actual: "+str(df_train_label[0:5].values))
print("pred:   "+str(predicted_values[24:35]))
