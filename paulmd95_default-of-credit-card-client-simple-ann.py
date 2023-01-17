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
#!/usr/bin/env python

#from __future__ import absolute_import, division, print_function

import numpy as np

import pandas as pd

import tensorflow

import keras

from sklearn.model_selection import train_test_split

from keras.models import Sequential

from keras.layers import Dense, Conv2D, Flatten

import tensorflow as tf

from sklearn.preprocessing import normalize

import seaborn

import matplotlib.pyplot as plt

from keras.optimizers import Adadelta

from keras.optimizers import Adam

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import accuracy_score,confusion_matrix,f1_score,precision_score

from keras import regularizers

from keras.layers.core import Dropout
data=pd.read_excel('../input/defcred/default of credit card clients.xls')

data.dropna(inplace=True)



print(data.shape)
data.drop_duplicates(inplace=True)

y=data.Y

y.drop(axis=1,index=['ID'],inplace=True)

from sklearn.preprocessing import LabelEncoder

encoder =  LabelEncoder()

y1 = encoder.fit_transform(y)

data.drop(axis=1,index=['ID'],inplace=True)
#normalise

scaler = StandardScaler()

scaler.fit(data)

data=scaler.transform(data)

#X=normalize(X, norm='l2', axis=1, copy=True, return_norm=False)

#print(X[:6])
#remove outliers

from scipy import stats

import numpy as np



z = np.abs(stats.zscore(data))

print(z)
threshold = 3

print((np.where(z > 3)))

X = data[(z < 3).all(axis=1)]

print(X.shape)

X=pd.DataFrame(X)

y=X[23]

from sklearn.preprocessing import LabelEncoder

encoder =  LabelEncoder()

y = encoder.fit_transform(y)

#print(X1.columns)

X.drop(X.columns[4:len(X.columns)-1], axis=1, inplace=True)

print(X.shape)

print(y)
print(data.shape)

X_plot=pd.DataFrame(data)

#X_plot.drop(axis=1,index=['ID'],inplace=True)

print(X_plot.head(5))


seaborn.pairplot(X_plot,vars=[1,2,3,4],hue=23)

plt.show()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#create model

model = Sequential()



#add model layers

model.add(Dense(32, activation='relu',input_shape=(5,)))

model.add(Dropout(0.5))

model.add(Dense(64, activation='tanh'))

model.add(Dropout(0.5))

model.add(Dense(128, activation='tanh'))

model.add(Dropout(0.5))

model.add(Dense(32, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(1, activation='sigmoid',kernel_regularizer=regularizers.l1_l2(0.1)))
#compile model using accuracy to measure model performance

opt = Adadelta()

model.compile(loss='mean_squared_error', optimizer=opt, metrics=['accuracy'])

model.summary()
#train the model

model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100,verbose=True)
p=model.predict_classes(X_test)

print(y_test)

y_test1=list(y_test)

#print(y_test[:100])

p=p.ravel()

p=list(p)
print(accuracy_score(y_test1,p))

print(precision_score(y_test1,p))

print(f1_score(y_test1,p))

print(confusion_matrix(y_test1,p))