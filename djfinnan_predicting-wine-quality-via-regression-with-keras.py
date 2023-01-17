from sklearn.metrics import mean_squared_error
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing 
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
import numpy as np
np.random.seed(16)
from keras import regularizers

#Loading the dataset
df = pd.read_csv("../input/red-wine-quality-cortez-et-al-2009/winequality-red.csv")

df = pd.get_dummies(df)

#Scaling the Dataset
df_scaled = preprocessing.scale(df)
df_scaled = pd.DataFrame(df_scaled, columns=df.columns)
df_scaled['quality'] = df['quality']
df= df_scaled


# Splitting the dataframe into a training and testing sets
X = df.loc[:, df.columns != 'quality'] 
y = df.loc[:, 'quality'] 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# Building the model
model=Sequential() 
model.add(Dense(252, activation= 'relu', input_dim=X_train.shape[1], kernel_regularizer=regularizers.l2(0.01), activity_regularizer=regularizers.l1(0.01)))
model.add(Dense(128, activation= 'relu',  kernel_regularizer=regularizers.l2(0.01), activity_regularizer=regularizers.l1(0.01)))
model.add(Dense(64, activation= 'relu',  kernel_regularizer=regularizers.l2(0.01), activity_regularizer=regularizers.l1(0.01)))
model.add(Dense(32, activation= 'relu',  kernel_regularizer=regularizers.l2(0.01), activity_regularizer=regularizers.l1(0.01)))
model.add(Dense(16, activation= 'relu',  kernel_regularizer=regularizers.l2(0.01), activity_regularizer=regularizers.l1(0.01)))
model.add(Dense(8, activation = 'relu',  kernel_regularizer=regularizers.l2(0.01), activity_regularizer=regularizers.l1(0.01)))
model.add(Dense(1))

#Compiling the model 
model.compile(loss='mse', optimizer='adam', metrics=['mse'])

#Training the model
model.fit(X_train, y_train, epochs=20)

#Results
train_pred = model.predict(X_train)
train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
test_pred = model.predict(X_test)
test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
print("Train RMSE: {:0.2f}".format(train_rmse))
print("Test RMSE: {:0.2f}".format(test_rmse))
