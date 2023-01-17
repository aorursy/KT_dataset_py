# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

#Load Libraries
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.ensemble import RandomForestClassifier
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow import keras
from keras.initializers import random_uniform
from sklearn import preprocessing
from tensorflow.keras.layers import Dense, Flatten, Activation, MaxPooling2D, Conv2D, Conv1D, MaxPooling1D, Convolution1D, Dropout
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
def feature_input_data(df,bResetIndex):
  if(bResetIndex):
    df = df.drop(['time'],axis=1).reset_index(drop=True)
  else:
    df = df.drop(['time'],axis=1)
  min_max_scaler = preprocessing.MinMaxScaler()
  df['signal'] = pd.DataFrame(min_max_scaler.fit_transform(df['signal'].values.reshape(-1,1)))
  temp = np.empty(len(df['signal']))
  p = df['signal'].values
  print(p)
  print(p.shape)
  for i in range(len(df['signal'])):
    if(i==0 or i==(p.shape[0]-1)):
      print(i)
      temp[i] = p[i]  
    else:
      temp[i] = np.abs(p[i+1] - p[i-1]) + p[i]    
  df['featured'] = temp
  return df

def get_split_train_val(df,split):
  #returns test, split
  return df[0:np.round(len(df)*split).astype(np.int32)],df[np.round(len(df)*split).astype(np.int32):-1]

def create_model(act_model):
  model = Sequential()
  model.add(Dense(100,  input_shape=input_shape, activation=act_model))
  model.add(Dropout(0.15))
  model.add(Dense(90,  input_shape=input_shape, activation=act_model))
  model.add(Dense(80, kernel_initializer=hidden_initializer, activation=act_model))
  model.add(keras.layers.LeakyReLU(alpha=0.3))
  model.add(Dense(80, kernel_initializer=hidden_initializer, activation=act_model,kernel_regularizer=tf.keras.regularizers.l1_l2(0.01)))
  model.add(tf.keras.layers.LSTM(40,activation=act_model))
  model.add(Dense(11, kernel_initializer=hidden_initializer, activation='softmax'))
  return model

def perform_model(model,optimiz,val_x,val_y,loss_s,batch_size=512):
  model.compile(optimizer = optimiz, loss=loss_s,metrics = ['acc'])
  model.fit(train_x,train_y,validation_data=(val_x, val_y),epochs=500,batch_size=batch_size)
  return model
#Load data
path_csv =  "/kaggle/input/clean-data/train_clean.csv"
data_raw = pd.read_csv(path_csv,sep=',')

#Balance the data
SEED = 42
hidden_initializer = random_uniform(seed=SEED)
from imblearn.over_sampling import SMOTE
smote = SMOTE('minority')
X_sm, Y_sm = smote.fit_sample(data_raw.drop('open_channels',axis=1),data_raw.pop('open_channels'))

dataset_x = pd.DataFrame({'time': X_sm.values[:, 0], 'signal': X_sm.values[:, 1]})
dataset_y = pd.DataFrame({'input_connections': Y_sm})
#create training and validation dataset
train_x,val_x = get_split_train_val(dataset_x,.8)
train_y,val_y = get_split_train_val(dataset_y,.8)

#Feature the data
train_x = feature_input_data(train_x,False)
val_x = feature_input_data(val_x,True)

#convert data for  input
train_x = train_x.values
train_y = train_y.values
val_x = val_x.values
val_y = val_y.values
train_x = train_x.reshape(train_x.shape[0],train_x.shape[1],1)
val_x = val_x.reshape(val_x.shape[0],val_x.shape[1],1)
input_shape = train_x.shape[1:3]
model = create_model('relu')
#Compile and fit model
adm = tf.optimizers.Adam(0.006)
rprop = tf.optimizers.RMSprop(0.006)
loss_s = 'sparse_categorical_crossentropy'
model = perform_model(model,rprop,val_x,val_y,loss_s,256)