# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



# Regression Example With Boston Dataset

# Import Data

import numpy as np

import pandas as pd

from pandas import read_csv

from keras.models import Sequential

from keras.layers import Dense

from keras.wrappers.scikit_learn import KerasRegressor

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import KFold

from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import Pipeline

# load dataset and split data 

dataset = pd.read_csv("../input/housing.csv", delim_whitespace=True, header=None)

X = dataset.iloc[:,0:13].values

Y = dataset.iloc[:,13].values
# define the model

def larger_model():

	# create model

	model = Sequential()

	model.add(Dense(16, input_dim=13, kernel_initializer='normal', activation='relu'))

	model.add(Dense(8, kernel_initializer='normal', activation='relu'))

	model.add(Dense(1, kernel_initializer='normal'))

	# Compile model

	model.compile(loss='mean_squared_error', optimizer='adam')

	return model
# fix random seed for reproducibility

seed = 7

np.random.seed(seed)

# evaluate model with standardized dataset

estimators = []

estimators.append(('standardize', StandardScaler()))

estimators.append(('mlp', KerasRegressor(build_fn=larger_model, epochs=50, batch_size=5, verbose=0)))

pipeline = Pipeline(estimators)

kfold = KFold(n_splits=10, random_state=seed)

results = cross_val_score(pipeline, X, Y, cv=kfold)

print("Results: mean %.2f ,sd %.2f MSE" % (results.mean(), results.std()))
