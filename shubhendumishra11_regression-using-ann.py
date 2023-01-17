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
!pip install tensorflow 

import matplotlib.pyplot as plt

import seaborn as sns

import tensorflow as tf

from tensorflow import keras

from tensorflow.keras import layers

from keras.models import Sequential

from keras.layers import Dense

from keras.wrappers.scikit_learn import KerasRegressor

from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score

data=pd.read_csv('../input/autompg-dataset/auto-mpg.csv')

data
data=data.drop(columns=['car name'])

data
origin=data.pop('origin')

data['uk']=(origin==1)*1.0

data['europe']=(origin==2)*2.0

data['usa']=(origin==3)*3.0

data
train_data=data.sample(frac=0.8, random_state=0)

test_data=data.drop(train_data.index)

print(train_data)

print(test_data)
train_data_state=train_data.describe()

train_data_state=train_data_state.transpose()

train_data_state

#train_data['horsepower'].describe()

a=train_data['horsepower']

a
a.unique()
a=a.replace('?',0)

a.unique()
def norm(x):

    return(x-train_data_state['mean'])/train_data_state['std']

normed_train_data=norm(train_data)

normed_train_data
normed_train_data=normed_train_data.drop(columns=['horsepower'])

normed_train_data
normed_train_data['horsepower']=a

normed_train_data
labels_of_train_data=normed_train_data['mpg']

features_of_train_data=normed_train_data.drop(columns=['mpg'])

print(labels_of_train_data)

print(features_of_train_data)
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(features_of_train_data,labels_of_train_data, test_size=0.3)
def build_model():

    model=keras.Sequential([layers.Dense(16, activation='relu'),layers.Dense(16, activation='relu'),

                                                 layers.Dense(1)])

    optimizer=tf.keras.optimizers.RMSprop(0.001)

    model.compile(loss='mse', optimizer=optimizer, metrics=['mae','mse'])

    return model
model=build_model()
#def baseline_model():

	# create model

#	model = Sequential()

#	model.add(Dense(13, input_dim=13, kernel_initializer='normal', activation='relu'))

#	model.add(Dense(1, kernel_initializer='normal'))

	# Compile model

#	model.compile(loss='mean_squared_error', optimizer='adam')

#	return model
 #evaluate model

#estimator = KerasRegressor(build_fn=baseline_model, epochs=100, batch_size=5, verbose=0)
#kfold = KFold(n_splits=10)

#results = cross_val_score(estimator, x_train, y_train, cv=kfold)

#print("Baseline: %.2f (%.2f) MSE" % (results.mean(), results.std()))
X=x_train.to_numpy(dtype=np.float)

Y=y_train.to_numpy(dtype=np.float)
model.fit(X,Y, epochs=1000)
X1=x_test.to_numpy(dtype=np.float)

Y1=y_test.to_numpy(dtype=np.float)
test_predictions=model.predict(X1)

test_predictions
loss,mae,mse=model.evaluate(X1,Y1,verbose=0)

print('Testing mean squared error:MPG', format(mse))