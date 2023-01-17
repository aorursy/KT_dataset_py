import numpy as np
import pandas as pd
import keras 
from keras.models import Sequential
from keras.layers import BatchNormalization, LSTM, Activation, Dense, Dropout, GRU,Conv1D,MaxPooling1D
from keras.initializers import Constant
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical
from keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
%matplotlib inline

import pickle
import itertools
import sqlite3
# need all of this to get reproducible results in Keras
# source: https://stackoverflow.com/questions/32419510/how-to-get-reproducible-results-in-keras
seed_value= 42

# 1. Set the `PYTHONHASHSEED` environment variable at a fixed value
import os
os.environ['PYTHONHASHSEED']=str(seed_value)

# 2. Set the `python` built-in pseudo-random generator at a fixed value
import random
random.seed(seed_value)

# 3. Set the `numpy` pseudo-random generator at a fixed value
import numpy as np
np.random.seed(seed_value)

# 4. Set the `tensorflow` pseudo-random generator at a fixed value
import tensorflow as tf
tf.set_random_seed(seed_value)

# 5. Configure a new global `tensorflow` session
from keras import backend as K
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)
df = pd.read_csv('../input/human-activity-data/DataSet.csv')
df
df
df['activity_label']=df.activity.replace({'Walking&party':0,'In bus':1,'Eat':2,'Walk':3,'Train':4,'At home':5,'In computer':6,'Video games':7,'In vehicle':8,'Picnic ':9,'Meeting':10,'Cooking':11})
test = df[df.activity.eq('Phone was out of the pocket (forgot)')] #I guess the final objective could be to try to guess what was that activity
columns = ['ACC_X','ACC_Y','ACC_Z','GYRO_X','GYRO_Y','GYRO_Z','EOG_L','EOG_R','EOG_H','EOG_V','activity']
train_mask = df.activity != 'Phone was out of the pocket (forgot)' 
X = df[columns][train_mask]
X = X.drop(['activity'],axis = 1)
Y = X.drop(['ACC_X','ACC_Y','ACC_Z','GYRO_X','GYRO_Y','GYRO_Z','EOG_L','EOG_R','EOG_H','EOG_V',],axis = 1)
X_train = X[:2468317]
X_test = X[2468317:] # 80/20 split
y_train = Y[:2468317]
y_test = Y[2468317:]
# from keras.utils import to_categorical
# y_train = to_categorical(y_train,3)
# # X_train = to_categorical(X_train,10)
# # X_train.shape

from sklearn.decomposition import PCA
pca=PCA(n_components=None)
X_train=pca.fit_transform(X_train)
X_test=pca.transform(X_test)
explained_variance=pca.explained_variance_ratio_
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)
from sklearn.preprocessing import LabelEncoder
encoder=LabelEncoder()
y_train=encoder.fit_transform(np.ravel(y_train))
y_train=pd.get_dummies(np.ravel(y_train))
from sklearn.preprocessing import LabelEncoder
encoder=LabelEncoder()
y_test=encoder.fit_transform(np.ravel(y_test))
y_test=pd.get_dummies(np.ravel(y_test))
from sklearn.decomposition import PCA
pca=PCA(n_components=None)
X_train=pca.fit_transform(X_train)
X_test=pca.transform(X_test)
explained_variance=pca.explained_variance_ratio_
explained_variance
from keras.models import Sequential
from keras.layers import Dense,Dropout
X_train.shape[1]
model=Sequential()
model.add(Dense(units=64,kernel_initializer='uniform',activation='relu',input_dim=X_train.shape[1]))

model.add(Dense(units=128,kernel_initializer='uniform',activation='relu'))

model.add(Dense(units=64,kernel_initializer='uniform',activation='relu'))

model.add(Dense(units=10,kernel_initializer='uniform',activation='softmax'))
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
history=model.fit(X_train,y_train,batch_size=256,epochs=22,validation_data=(X_test,y_test))

df.to_csv('DataSet.csv')