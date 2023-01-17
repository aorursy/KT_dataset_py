import numpy as np

import pandas as pd 

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler

import scipy as sp 

import sklearn

import random 

import time 

from sklearn import preprocessing, model_selection

from keras.models import Sequential 

from keras.layers import Dense 

from keras.utils import np_utils

from sklearn.preprocessing import LabelEncoder

from keras.utils.np_utils import to_categorical

from sklearn.utils import shuffle

from sklearn.utils import class_weight
df_train1 = pd.read_csv('/kaggle/input/bitsf312-lab1/train.csv')

df_test1 = pd.read_csv('/kaggle/input/bitsf312-lab1/test.csv')
df_train = df_train1.drop(['ID'],axis=1).copy()

df_test = df_test1.drop(['ID'],axis=1).copy()

df_train['Size'].replace({'?':'Medium'},inplace=True)
#FOR TRAINING SET

one_hot_tr = pd.get_dummies(df_train['Size'])

df_train = df_train.drop('Size',axis = 1)

df_train = df_train.join(one_hot_tr)

#FOR TESTING SET

one_hot_te = pd.get_dummies(df_test['Size'])

df_test = df_test.drop('Size',axis = 1)

df_test = df_test.join(one_hot_te)
df_train['Number of Quantities'].value_counts()

df_train['Number of Quantities'].replace({'?':2},inplace=True)

df_train['Number of Insignificant Quantities'].value_counts()

df_train['Number of Insignificant Quantities'].replace({'?':0},inplace=True)

df_train['Number of Special Characters'].value_counts()

df_train['Number of Special Characters'].replace({'?':0},inplace=True)

df_train['Total Number of Words'].value_counts()

df_train['Total Number of Words'].replace({'?':20},inplace=True)

df_train['Difficulty'].replace({'?':5},inplace=True)
import seaborn as sns

df_train.corr()
X_train = df_train.drop(['Class'], axis = 1)

Y_train = df_train['Class']

y = df_train['Class'].copy()

X_test = df_test

Y_train = np_utils.to_categorical(Y_train)

X_test = np.array(X_test)

X_test = np.array(X_test)
train_x, test_x, train_y, test_y = model_selection.train_test_split(X_train,Y_train,test_size = 0.2, random_state = 42)
class_weights = class_weight.compute_class_weight('balanced',np.unique(y),y)
from keras.regularizers import l2

input_dim = len(X_train.columns) 

from keras.layers import Dropout

from keras import optimizers



#MODEL STRUCTURE

model2 = Sequential()

model2.add(Dense(13, input_dim = input_dim , activation = 'relu'))

model2.add(Dense(16, activation = 'relu'))

model2.add(Dense(10, activation = 'relu'))

model2.add(Dropout(0.1))

model2.add(Dense(10, activation = 'relu'))

model2.add(Dropout(0.1))

model2.add(Dense(6, activation = 'softmax'))



#MODEL COMPILATION

model2.compile(loss = 'categorical_crossentropy' , optimizer = 'adam' , metrics = ['accuracy'] )



#MODEL TRAIN

model2.fit(train_x, train_y, epochs = 100, batch_size = 2, class_weight=class_weights)



#MODEL EVALUATION

scores = model2.evaluate(test_x, test_y)

print("\n%s: %.2f%%" % (model2.metrics_names[1], scores[1]*100))
z = model2.predict_classes(X_test)

out_id = df_test1['ID'].copy()

out_df_ker = pd.DataFrame(columns = ['ID','Class'])

out_df_ker['ID'] = out_id

out_df_ker['Class'] = z

out_df_ker.head()

out_df_ker.to_csv('One_Hot_L2.csv', index=False)