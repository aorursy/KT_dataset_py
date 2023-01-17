#Importing the Libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
#Importing the Dataset

house_data = pd.read_csv('../input/kc_house_data.csv')
#Let's see how many rows and columns of data are available in the dataset

house_data
#Looking at the top 10 rows of the dataset

house_data.head(10)
#Looking at the bottom 10 of the dataset

house_data.tail(10)
house_data.describe()
house_data.info()
house_data
house_data.columns
plt.figure(figsize=(20, 10))

sns.scatterplot(x = house_data['price'], y = house_data['sqft_living'], color = 'g')
plt.figure(figsize=(20, 10))

sns.scatterplot(x = house_data['price'], y = house_data['sqft_lot'], color = 'g')
house_data.hist(bins=20, figsize=(20, 20), color = 'g')
plt.figure(figsize=(20, 20))

sns.heatmap(house_data.corr(), annot=True)
sns.pairplot(house_data)
house_data_important = house_data[ ['price', 'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'sqft_above', 'sqft_basement', 'yr_built']   ]
house_data_important
sns.pairplot(house_data_important)
#We will use only the important features that we set in the Visualization part.

X = house_data_important.drop(['price'], axis =1)

y = house_data['price']
#Let's take a look at X

X.head(10)
X.shape
#and y

y.head(10)
y.shape
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

X_scaled = scaler.fit_transform(X)
X_scaled
# Scaling y

y = y.values.reshape(-1, 1)
y_scaled = scaler.fit_transform(y)
y_scaled
#We will be splitting the data into 75% for training and 25% for testing.



from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size = 0.25)
print("We have", X_train.shape, "for training and", X_test.shape, "for testing.")
#We will use keras in building our ANN Model. Let's start by importing the needed libraries

import keras

from keras.models import Sequential

from keras.layers import Dense
#Lets start building the model.

model = Sequential()

model.add(Dense(32, input_dim = 7, activation = 'relu'))

model.add(Dense(64, activation = 'relu'))

model.add(Dense(128, activation = 'relu'))

model.add(Dense(1, activation = 'linear'))

model.compile(loss = 'mean_squared_error', optimizer = 'Adam')

model.summary()
#We will train the model and store all the history of the training in the variable history_epoch

history_epoch = model.fit(X_train, y_train, batch_size = 32, epochs = 50, validation_split=0.2)
#First lets visualize how the loss in the training differs from the validation

history_epoch.history.keys()
plt.plot(history_epoch.history['loss'])

plt.plot(history_epoch.history['val_loss'])

plt.title('Model Loss During Training')

plt.xlabel('Epoch')

plt.ylabel('Training and Validation Loss')

plt.legend(['Training Loss', 'Validation Loss'])
#Let's do the prediction for the test data

y_predictions = model.predict(X_test)
#Transform back y_predict and y_test into its original values.

orig_y_test = scaler.inverse_transform(y_test)

orig_y_predict = scaler.inverse_transform(y_predictions)
n = len(X_test)

k = X_test.shape[1]
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from math import sqrt



RMSE = float(format(np.sqrt(mean_squared_error(orig_y_test, orig_y_predict)),'.3f'))

MSE = mean_squared_error(orig_y_test, orig_y_predict)

MAE = mean_absolute_error(orig_y_test, orig_y_predict)

r2 = r2_score(orig_y_test, orig_y_predict)

adj_r2 = 1-(1-r2)*(n-1)/(n-k-1)



print('RMSE =',RMSE, '\nMSE =',MSE, '\nMAE =',MAE, '\nR2 =', r2, '\nAdjusted R2 =', adj_r2) 

#We will be using more features

more_features = ['bedrooms','bathrooms','sqft_living','sqft_lot','floors', 'sqft_above', 'sqft_basement', 'waterfront', 'view', 'condition', 'grade', 'sqft_above', 'yr_built', 

'yr_renovated', 'zipcode', 'lat', 'long', 'sqft_living15', 'sqft_lot15']



X_2 = house_data[more_features]
#Scaling

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

X_2_scaled = scaler.fit_transform(X_2)
y_2 = house_data['price']

y_2 = y_2.values.reshape(-1,1)

y_2_scaled = scaler.fit_transform(y_2)
#Splitting the dataset into training and testing

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_2_scaled, y_2_scaled, test_size = 0.25)
#Lets start building the model again and 1 more layer.

model = Sequential()

model.add(Dense(32, input_dim = 19, activation = 'relu'))

model.add(Dense(64, activation = 'relu'))

model.add(Dense(128, activation = 'relu'))

model.add(Dense(256, activation = 'relu'))

model.add(Dense(1, activation = 'linear'))

model.compile(loss = 'mean_squared_error', optimizer = 'Adam')

model.summary()
history_epoch_2 = model.fit(X_train, y_train, batch_size = 32, epochs = 50, validation_split=0.2)
#Visualizing the losses

plt.plot(history_epoch_2.history['loss'])

plt.plot(history_epoch_2.history['val_loss'])

plt.title('Model Loss During Training')

plt.ylabel('Training and Validation Loss')

plt.xlabel('Epoch number')

plt.legend(['Training Loss', 'Validation Loss'])
#Let's do the prediction for the test data

y_predictions_2 = model.predict(X_test)
#Transform back y_predict and y_test into its original values.

orig_y_test = scaler.inverse_transform(y_test)

orig_y_predict_2 = scaler.inverse_transform(y_predictions_2)
n = len(X_test)

k = X_test.shape[1]



from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from math import sqrt



RMSE = float(format(np.sqrt(mean_squared_error(orig_y_test, orig_y_predict_2)),'.3f'))

MSE = mean_squared_error(orig_y_test, orig_y_predict_2)

MAE = mean_absolute_error(orig_y_test, orig_y_predict_2)

r2 = r2_score(orig_y_test, orig_y_predict_2)

adj_r2 = 1-(1-r2)*(n-1)/(n-k-1)



print('RMSE =',RMSE, '\nMSE =',MSE, '\nMAE =',MAE, '\nR2 =', r2, '\nAdjusted R2 =', adj_r2) 
