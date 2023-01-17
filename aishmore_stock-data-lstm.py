# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#For data processing

import pandas as pd

import numpy as np

import datetime as dt



#For data visualization

import matplotlib.pyplot as plt

import seaborn as sns



#For training and evaluating models 

from sklearn import metrics

from sklearn.preprocessing import MinMaxScaler

from keras.models import Sequential

from keras.layers import Dense

from keras.layers import LSTM

from keras.layers import Dropout
df = pd.read_csv('../input/price-volume-data-for-all-us-stocks-etfs/Stocks/gs.us.txt')

df.head()
#Convert date to datetime value

df['Date'] = pd.to_datetime(df['Date'],format='%Y-%m-%d')
df.info()
plt.rcParams['figure.figsize'] = (17,6)

plt.grid(True)

plt.plot(df['Date'],df['Close'], color='b')

plt.yticks(np.arange(0,350, 50))

plt.xlabel('Date')

plt.ylabel('Close Price')

plt.title('Close Prices for Goldman Sachs Shares over time')

plt.show()
#Set datetime as the index value

dt = df.copy()

dt = dt.set_index(dt['Date'])

dt = dt.drop(columns=['Date','High','Low','Open','Volume','OpenInt'], axis=1)

dt.head()
#Scale all of the data before splitting

sc = MinMaxScaler(feature_range=(0,1))

dt = sc.fit_transform(np.array(dt).reshape(-1,1))

print(df.shape)

print(dt.shape)
#Visualization of the training-testing data split

test = df[int(len(df)*0.85):]

ax = df['Close'].plot(figsize = (17,6),color='blue', label='Training Data')

ax = test['Close'].plot(figsize = (17,6),color='red', label='Testing Data')

ax.set(xlabel='Dates', ylabel='Daily sales')

ax.legend(loc='lower right')

plt.grid(True)

plt.show()
#Split data into training and testing sets

total_size = len(dt)

train_size = int(total_size * 0.85)

test_size = total_size - train_size

train_set, test_set = dt[0:train_size,:], dt[train_size:total_size, :1]

print(train_set.shape, test_set.shape)
#convert arrays into matrix

def create_dataset(data, timestep):

    dataX = []

    dataY = []

    numdata = len(data)-timestep-1

    for i in range(numdata):

        val_X = data[i:(i+timestep),0]

        val_Y = data[i+timestep,0]

        dataX.append(val_X)

        dataY.append(val_Y)

    X_arr = np.array(dataX)

    y_arr = np.array(dataY)

    return X_arr, y_arr
#reshape and split our train and test data into X and y

#selecting same timestep for both train and test sets

timestep = 50

X_train, y_train = create_dataset(train_set, timestep)

X_test, y_test = create_dataset(test_set, timestep)



print(X_train.shape)

print(y_train.shape)

print(X_test.shape)

print(y_test.shape)
#Reshape data as 3D array which is required for LSTM

X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)

X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)



print(X_train.shape)

print(y_train.shape)

print(X_test.shape)

print(y_test.shape)
#Initialize model

model = Sequential()



#Adding first LSTM layer and dropout regularisation

model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))

model.add(Dropout(0.15))



#Adding second LSTM layer

model.add(LSTM(units=50, return_sequences=True))



#Adding third LSTM layer

model.add(LSTM(units=50))



#Output Layer

model.add(Dense(units=1))
#Compile the model

model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'] )
#Get summary of the model details

model.summary()
history = model.fit(X_train, y_train, validation_split=0.20, epochs=50, verbose=1)
#Make predictions using the model

train_predict = model.predict(X_train)

test_predict = model.predict(X_test)
#Transform data to original form (unscaled values)

y_train_pred = sc.inverse_transform(train_predict)

y_test_pred = sc.inverse_transform(test_predict)
#Finding error produced by model

mae = metrics.mean_absolute_error(y_test, y_test_pred)

mse = metrics.mean_squared_error(y_test,y_test_pred)

rmse = np.sqrt(mse)



print('Mean Absolute Error:', mae)

print('Mean Squared Error:', mse)

print('Root Mean Squared Error:', rmse)
timestep = 50 #since we took the timestep as 50

#adjust train predictions plot

trainPredPlot = np.empty_like(dt)

trainPredPlot[:,:] = np.nan

trainPredPlot[timestep:len(y_train_pred)+timestep,:] = y_train_pred

#adjust test predictions plot

testPredPlot = np.empty_like(dt)

testPredPlot[:,:] = np.nan

testPredPlot[len(y_train_pred)+(2*timestep)+1:len(dt)-1,:] = y_test_pred
#plot predictions

plt.rcParams['figure.figsize'] = (17,6)

dt_plot = sc.inverse_transform(dt)

plt.plot(dt_plot,color='black', label='Actual Data')

plt.plot(trainPredPlot,color='blue', label='Predict Train Data')

plt.plot(testPredPlot,color='red', label='Predict Test Data')

plt.legend()

plt.grid(True)

plt.show()
#closer look at test predictions

plt.rcParams['figure.figsize'] = (18,7)

dt_plot = sc.inverse_transform(dt)

plt.plot(dt_plot,color='black', label='Actual Data')

plt.plot(testPredPlot,color='red', label='Predict Test Data')

plt.xlim(4000,4700)

plt.ylim(100,300)

plt.legend()

plt.grid(True)

plt.show()
# Plot training & validation accuracy values

plt.rcParams['figure.figsize'] = (15, 6)



plt.plot(history.history['accuracy'])

plt.plot(history.history['val_accuracy'])

plt.title('Model accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')

plt.show()



# Plot training & validation loss values

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('Model loss')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')

plt.show()