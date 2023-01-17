import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from keras.models import Sequential

from keras.layers import Dense

from keras.wrappers.scikit_learn import KerasRegressor

from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import Pipeline

from sklearn.model_selection import train_test_split

import gc

import pickle

import sys

from sklearn.metrics import mean_absolute_error, mean_squared_error 
pkl_file = open('/kaggle/input/ashrae-preprocessing-train/data_train.pkl', 'rb')

data_train = pickle.load(pkl_file)

pkl_file.close()

pkl_file = open('/kaggle/input/ashrae-preprocessing-train/y.pkl', 'rb')

y = pickle.load(pkl_file)

pkl_file.close()



X, testX, Y, testy = train_test_split(data_train, y, train_size=0.8, test_size=0.2)

# We take the log (because the evaluation metric is RMSLE and we'll use the RMSE metric for training)

Y = np.log1p(Y)
# define base model

def baseline_model():

    # create model

    model = Sequential()

    model.add(Dense(13, input_dim=13, kernel_initializer='normal', activation='relu'))

    model.add(Dense(1, kernel_initializer='normal'))

    # Compile model

    model.compile(loss='mean_squared_error', optimizer='adam')

    return model
estimators = []

estimators.append(('standardize', StandardScaler()))

estimators.append(('mlp', KerasRegressor(build_fn=baseline_model, epochs=50, batch_size=10000, verbose=1)))

pipeline = Pipeline(estimators)

pipeline.fit(X, Y)
print(mean_absolute_error(testy, np.expm1(pipeline.predict(testX))))