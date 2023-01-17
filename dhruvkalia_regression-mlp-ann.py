# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from sklearn.impute import SimpleImputer

from sklearn.model_selection import train_test_split

import keras as keras

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import r2_score

from sklearn.preprocessing import MinMaxScaler

from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer

from sklearn.model_selection import train_test_split

from tensorflow.python import keras

from keras import layers

import tensorflow as tf

from scipy.stats import reciprocal

from sklearn.model_selection import RandomizedSearchCV



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# Importing data

df = pd.read_csv("../input/calcofi/bottle.csv")

df = df.loc[1:1000, ['T_degC', 'Salnty', 'Depthm']]

df.head()
# Creating train and test dataset

msk = np.random.rand(len(df)) < 0.8

train_raw = df[msk]

test_raw = df[~msk]
# Fixing missing values issue and imputing

imputer = SimpleImputer()

train = pd.DataFrame(imputer.fit_transform(train_raw))

train.columns = df.columns

train.rename(columns={'T_degC': 'TEMP', 'Salnty': 'SALINITY', 'Depthm': 'DEPTH'}, inplace=True)

train = train.reindex(columns={'SALINITY', 'DEPTH', 'TEMP'})

print(train.head())
train_x = train[['SALINITY', 'DEPTH']]

train_y = train[['TEMP']]
# Splitting into train and test dataset

X_train_full, X_test, y_train_full, y_test = train_test_split(train_x, train_y, random_state=0)

X_train, X_valid, y_train, y_valid = train_test_split(X_train_full, y_train_full, random_state=1)
# Feature Scaling

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_valid = scaler.transform(X_valid)

X_test = scaler.transform(X_test)


def build_model(n_hidden=1, n_neurons=30, learning_rate=3e-3, input_shape=[2]):

    model = keras.models.Sequential()

    model.add(keras.layers.InputLayer(input_shape=input_shape))

    for layer in range(n_hidden):

        model.add(keras.layers.Dense(n_neurons, activation="relu"))

    model.add(keras.layers.Dense(1))

    optimizer = tf.keras.optimizers.SGD(lr=learning_rate)

    model.compile(loss="mse", optimizer=optimizer)

    return model


keras_reg = keras.wrappers.scikit_learn.KerasRegressor(build_model)

param_distribs = {

    "n_hidden": [0,1,2,3],

    "n_neurons": np.arange(1,100),

    "learning_rate": reciprocal(3e-4, 3e-2),

}



rnd_search_cv = RandomizedSearchCV(keras_reg, param_distribs, n_iter=10, cv=3)

rnd_search_cv.fit(X_train, y_train, epochs=100,

                 validation_data=(X_valid, y_valid),

                 callbacks=[keras.callbacks.EarlyStopping(patience=10)])
rnd_search_cv.best_params_
model = build_model(n_hidden=rnd_search_cv.best_params_['n_hidden'], n_neurons=rnd_search_cv.best_params_['n_neurons'], learning_rate=rnd_search_cv.best_params_['learning_rate'], input_shape=[2])
model.summary()
# # Creating model

# model = keras.models.Sequential([

#     keras.layers.Dense(20, activation="relu"),

#     keras.layers.Dense(1)

# ])

# model.compile(loss="mean_squared_error", optimizer="sgd")
# Fitting

history = model.fit(X_train, y_train, epochs=20, validation_data=(X_valid, y_valid))
# Evaluating

mse_test = model.evaluate(X_test, y_test)
# # Predicting

pred = model.predict(X_test)

# Accuracy measures

print("Mean absolute error: %.2f" % np.mean(np.absolute(pred - np.asanyarray(y_test))))

print("Residual sum of squares (MSE): %.2f" % np.mean((pred - np.asanyarray(y_test)) ** 2))

print("R2-score: %.2f" % r2_score(pred, np.asanyarray(y_test)))
# # Plotting

plt.scatter(pd.DataFrame(X_test[:,0]), y_test, color='blue')

plt.scatter(pd.DataFrame(X_test[:,0]), pred, color='red')

plt.title('Truth or Bluff (Random Forest Regression)')

plt.xlabel("Salinity")

plt.ylabel("Temparature")

plt.show()