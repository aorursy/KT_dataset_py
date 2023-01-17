# packages for data processing and manipulation



!pip install PyDrive # for loading csv from Drive



from google.colab import files

import io



import pandas as pd

import numpy as np



import matplotlib.pyplot as plt

import seaborn as sns





# for loading csv from Drive 1. according to https://medium.freecodecamp.org/how-to-transfer-large-files-to-google-colab-and-remote-jupyter-notebooks-26ca252892fa

import os

from pydrive.auth import GoogleAuth

from pydrive.drive import GoogleDrive

from google.colab import auth

from oauth2client.client import GoogleCredentials
# for loading csv from Drive 2.

auth.authenticate_user()

gauth = GoogleAuth()

gauth.credentials = GoogleCredentials.get_application_default()

drive = GoogleDrive(gauth)
# packages for estimation



from sklearn.preprocessing import normalize



from sklearn.model_selection import KFold

from sklearn.model_selection import train_test_split



from sklearn.neural_network import MLPRegressor

from sklearn.metrics import mean_squared_error



import keras

from keras.models import Sequential

from keras.layers import Dense, Activation, Dropout

from keras.callbacks import EarlyStopping

from keras.layers.normalization import BatchNormalization



from sklearn.ensemble import RandomForestRegressor
download = drive.CreateFile({'id': '1c_8x-EIqzAyBJz8e29D4Xnhesqsz7zFV'}) # ID is the name of the file you are using. in my case it was automatically renamed.

download.GetContentFile('train.csv')
download = drive.CreateFile({'id': '1bDQIR7nt9KjLrQAdHY7B3By17H8FxR0j'})

download.GetContentFile('test.csv')
#uploaded = files.upload()
# loading training set

data_train = pd.read_csv("train.csv")

data_train.set_index("id", inplace = True)
# loading test set

data_test = pd.read_csv("test.csv")

data_test.set_index("id", inplace = True)
data_train.head()
data_test.head()
data_train.info()
# looking for missing values

plt.figure(figsize=(15,8))

sns.heatmap(data_train.isnull(), cbar=False)
# Splitting datasets to independent and dependent variables



indep_labels = ["ra", "dec", "u", "g", "r", "i", "z", "size", "ellipticity"]

dep_label = ["redshift"]



data_train_X = data_train[indep_labels]

data_train_Y = data_train[dep_label]



data_test_X = data_test[indep_labels]

#data_test_Y = data_test[dep_label] # because there is no redshift label in the test dataset, there is no data_test_Y given, we have to give an estimate for it.
# Datasets

data_train_X = data_train[indep_labels]

data_train_Y = data_train[dep_label]



# Train/test split

train_X, valid_X, train_Y, valid_Y = train_test_split(data_train_X, data_train_Y, train_size=0.8, random_state=0)
import time

start = time.time()



# Multi-layer perceptron

MLP_reg = MLPRegressor(hidden_layer_sizes=(200,100),

                       activation="relu",

                       solver="adam",

                       alpha=0.0001, batch_size="auto",

                       learning_rate="constant", learning_rate_init=0.001,

                       power_t=0.5, max_iter=200, shuffle=True,

                       random_state=None, tol=0.0001, verbose=True,

                       warm_start=False, momentum=0.9, nesterovs_momentum=True,

                       early_stopping=True, validation_fraction=0.1,

                       beta_1=0.9, beta_2=0.999, epsilon=1e-08, n_iter_no_change=10)



MLP_reg_fit = MLP_reg.fit(data_train_X, data_train_Y)



valid_y_pred_MLP = MLP_reg_fit.predict(data_test_X)



MSE_valid_MLP = mean_squared_error(valid_y_pred_MLP, valid_Y)



end = time.time()

print(end - start)
MSE_valid_MLP
y_pred = MLP_reg_fit.predict(data_test_X)
y_pred
predtosave = pd.DataFrame(y_pred, columns=["redshift"])
predtosave.to_csv('katona_sandor_predtoredshift.csv', index_label="id", header=True )
files.download('katona_sandor_predtoredshift.csv')
# Datasets



# Splitting datasets to independent and dependent variables



indep_labels = ["ra", "dec", "u", "g", "r", "i", "z", "size", "ellipticity"]

dep_label = ["redshift"]



data_train_X = data_train[indep_labels]

data_train_Y = data_train[dep_label]



data_test_X = data_test[indep_labels]
# Normalisation of the data_train_X



data_train_Xn = (data_train_X - data_train_X.mean()) / data_train_X.std() # not used in the model down
# Creating the model



model = Sequential()



# number of variables in training data

n_cols = data_train_Xn.shape[1]



# adding model layers

model.add(Dense(200, input_shape=(n_cols,)))

model.add(Activation('relu'))



model.add(Dense(200))

model.add(Activation('relu'))



model.add(Dense(200))

model.add(Activation('relu'))



model.add(Dense(200))

model.add(Activation('relu'))



model.add(Dense(1))
# Compiling the model



model.compile(optimizer='adam', loss='mean_squared_error')
# Training the model with the possibility of early stopping (MSE is shown here!)



import time

start = time.time()



early_stopping_monitor = EarlyStopping(patience=3)



model.fit(data_train_X, data_train_Y, validation_split=0.2, batch_size=200, epochs=15, callbacks=[early_stopping_monitor])





end = time.time()

print(end - start)
y_pred_keras = model.predict(data_test_X)
y_pred_keras
predtosave_keras = pd.DataFrame(y_pred_keras, columns=["redshift"])
predtosave_keras.to_csv('katona_sandor_predtoredshift_keras.csv', index_label="id", header=True )
files.download('katona_sandor_predtoredshift_keras.csv')
# Datasets



# Splitting datasets to independent and dependent variables



indep_labels = ["ra", "dec", "u", "g", "r", "i", "z", "size", "ellipticity"]

dep_label = ["redshift"]



data_train_X = data_train[indep_labels]

data_train_Y = data_train[dep_label]



# Train/test split

train_X, valid_X, train_Y, valid_Y = train_test_split(data_train_X, data_train_Y, train_size=0.8, random_state=0)
import time

start = time.time()



# Random forest regression

rf = RandomForestRegressor(n_estimators=190, criterion="mse",

                           max_depth=None, min_samples_split=2,

                           min_samples_leaf=1, min_weight_fraction_leaf=0.0,

                           max_features="auto", max_leaf_nodes=None,

                           min_impurity_decrease=0.0, min_impurity_split=None,

                           bootstrap=True, oob_score=False, n_jobs=-1,

                           random_state=None, verbose=0, warm_start=False)



rf_fit = rf.fit(train_X, train_Y)



valid_y_pred_rf = rf_fit.predict(valid_X)



MSE_valid_rf = mean_squared_error(valid_y_pred_rf, valid_Y)



end = time.time()

print(end - start)
MSE_valid_rf
y_pred_rf = rf_fit.predict(data_test_X)
y_pred_rf
predtosave_rf = pd.DataFrame(y_pred_rf, columns=["redshift"])
predtosave_rf.to_csv('katona_sandor_predtoredshift.csv', index_label="id", header=True )
files.download('katona_sandor_predtoredshift.csv')