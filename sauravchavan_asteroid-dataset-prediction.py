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
import keras as K

from keras.models import Sequential

from keras.layers import Dense

from keras.wrappers.scikit_learn import KerasClassifier

from sklearn.model_selection import cross_val_score

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import StratifiedKFold

from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import Pipeline



import sklearn as sk

from sklearn.linear_model import LogisticRegression

import pandas as pd

import os



import tensorflow as tf
data = pd.read_csv("/kaggle/input/asteroid-dataset/dataset.csv")
data
data.info()
data.columns
y_data = data["pha"].astype(str) 

y_data
data_new = data.drop(['id', 'spkid', 'full_name', 'pdes', 'name', 'prefix', 'neo', 'pha', 'orbit_id', 'equinox', 'class',],axis=1)
X_data = data_new

X_data
X_data = X_data.fillna(0)

y_data = y_data.fillna(0)
encoder = LabelEncoder()

encoder.fit(y_data)

encoded_Y = encoder.transform(y_data)
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_data,encoded_Y,random_state=0)

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
# def create_baseline():

# 	# create model

# 	model = Sequential()

# 	model.add(Dense(60, input_dim=60, activation='relu'))

# 	model.add(Dense(1, activation='sigmoid'))

# 	# Compile model

# 	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# 	return model

# # evaluate model with standardized dataset

# estimator = KerasClassifier(build_fn=create_baseline, epochs=100, batch_size=5, verbose=0)

# kfold = StratifiedKFold(n_splits=10, shuffle=True)

# results = cross_val_score(estimator, X_train, y_train, cv=kfold)

# print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
from sklearn import preprocessing

scaler = preprocessing.StandardScaler().fit(X_train)

scaler.transform(X_train)

LR = LogisticRegression(random_state=0, solver='lbfgs', multi_class='auto',max_iter=1000).fit(X_train, y_train)

LR.predict(X_test)

print(LR.score(X_test,y_test))
my_init = K.initializers.glorot_uniform(seed=1)

model = K.models.Sequential()

model.add(K.layers.Dense(units=34, input_shape = (718893, 34),

  activation='tanh', kernel_initializer=my_init)) 

model.add(K.layers.Dense(units=17, activation='tanh',

  kernel_initializer=my_init)) 

model.add(K.layers.Dense(units=8, activation='tanh',

  kernel_initializer=my_init)) 

model.add(K.layers.Dense(units=1, activation='sigmoid',

  kernel_initializer=my_init))

simple_sgd = K.optimizers.SGD(lr=0.01)  

model.compile(loss='binary_crossentropy',

  optimizer=simple_sgd, metrics=['accuracy'])  
ACCURACY_THRESHOLD=0.9765
# class myCallback(tf.keras.callbacks.Callback):

#     def on_epoch_end(self,epoch,logs={}):

#         if(logs.get('accuracy') > ACCURACY_THRESHOLD):

#             print("\nReached {} accuracy so cancelling!".format(ACCURACY_THRESHOLD))

#             self.model.stop_training=True

# callbacks = myCallback()
max_epochs = 5

# my_logger = MyLogger(n=50)

h = model.fit(X_train, y_train, batch_size=32,

  epochs=max_epochs, verbose=0)
np.set_printoptions(precision=4, suppress=True)

eval_results = model.evaluate(X_test, y_test, verbose=0) 

print("\nLoss, accuracy on test data: ")

print("%0.4f %0.2f%%" % (eval_results[0], \

  eval_results[1]*100))