import numpy as np   ## mathmetical operation library

import pandas as pd

import tensorflow as tf

import sklearn

import matplotlib.pyplot as plt
data = pd.read_csv('../input/heart_deases_data.csv',delim_whitespace=True)
data
data.columns
corelation = data.corr()
corelation
import seaborn as sns
sns.heatmap(corelation)
feature = data.drop('age',axis=1)
feature = feature.drop('class',axis=1)
feature.head()
target = data[['age','class']]
target.head()
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(feature,target,test_size = .2)
x_train.head()
x_test.head()
y_train.head()
y_test.head()
n_col = x_train.shape[1]
from keras.layers import Input

from keras.utils import plot_model

from keras.models import Model

from keras.layers import Input

from keras.layers import Dense

visible = Input(shape=(n_col,))

hidden1 = Dense(100, activation='relu')(visible)

hidden2 = Dense(200, activation='relu')(hidden1)

hidden3 = Dense(100, activation='relu')(hidden2)

hidden4 = Dense(100, activation='relu')(hidden3)

hidden5 = Dense(100, activation='relu')(hidden4)

hidden6 = Dense(100, activation='relu')(hidden5)

hidden7 = Dense(100, activation='relu')(hidden6)

output = Dense(2)(hidden7)

model = Model(inputs=visible, outputs=output)

model.compile(optimizer='adam',loss='mean_absolute_error')
model.summary()
plot_model(model)
model.fit(x_train,y_train,epochs=500)
model.get_weights()
model.evaluate(x_test,y_test)
model.predict(x_test)