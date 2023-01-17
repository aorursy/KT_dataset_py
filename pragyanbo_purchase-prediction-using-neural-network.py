import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt

import statistics

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split

from sklearn.metrics import r2_score

from sklearn.metrics import mean_absolute_error

from sklearn.metrics import mean_squared_error

import tqdm

import pickle

import matplotlib.pyplot as plt

from tensorflow.keras.callbacks import EarlyStopping

import tensorflow

from tensorflow import keras

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense
dataset = pd.read_csv('../input/BlackFriday.csv')
dataset.isnull().sum()



xdf = dataset.iloc[:,0:11]

ydf = dataset.iloc[:,-1]



xdf['Product_Category_2'].fillna(0,inplace = True)

xdf['Product_Category_3'].fillna(0,inplace = True)



xdf["Product_Category_2"] = xdf["Product_Category_2"].astype(int)

xdf["Product_Category_3"] = xdf["Product_Category_3"].astype(int)

#delete unnecesary columns like userid and prodid

xdf = xdf.drop(xdf.columns[[0, 1]], axis=1)

#Encoding Age

labelEncoderAge = LabelEncoder()

xdf.iloc[:,1] = labelEncoderAge.fit_transform(xdf.iloc[:,1])



#Encoding gender

labelEncoderGender = LabelEncoder()

xdf.iloc[:,0] = labelEncoderGender.fit_transform(xdf.iloc[:,0])



#Encoding City category

labelEncoderCity = LabelEncoder()

xdf.iloc[:,3] = labelEncoderCity.fit_transform(xdf.iloc[:,3])



#Encoding current city yr

labelEncoderCityyr = LabelEncoder()

xdf.iloc[:,4] = labelEncoderCityyr.fit_transform(xdf.iloc[:,4])

X = xdf.iloc[:,:].values

Y = ydf.values

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 42)#42
y_train = np.array(y_train)

y_test = np.array(y_test)

y_train = y_train.ravel()

y_test = y_test.ravel()
# Defining a Neural Network Model



model = Sequential()

model.add(Dense(900, activation='relu', input_dim=9))

model.add(Dense(450, activation='relu'))

model.add(Dense(450, activation='relu'))

model.add(Dense(1))



model.summary()



# Compile

model.compile(loss='mean_squared_error', 

              optimizer='adam', 

              metrics=['mean_squared_error', 'mean_absolute_percentage_error'])
# Early stopping while fitting model

# early_stopping_monitor = EarlyStopping(patience=3)

history = model.fit(X_train, y_train, epochs=16, validation_data=(X_test, y_test), verbose=1)
train_pred = model.predict(X_train)

train_score = np.sqrt(mean_squared_error(y_train, train_pred))

print('Train ANN RMSE:', train_score)
y_test_dt=model.predict(X_test)

test_dt_rmse = np.sqrt(mean_squared_error(y_test, y_test_dt))

print("Test ANN RMSE",test_dt_rmse)