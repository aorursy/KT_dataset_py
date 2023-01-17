import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import os
df = pd.read_csv('/kaggle/input/waves-measuring-buoys-data-mooloolaba/Coastal Data System - Waves (Mooloolaba) 01-2017 to 06 - 2019.csv')

df.head()
# Deleting NaN values

df.replace(-99.90, np.nan, inplace=True)

df.drop('Date/Time', axis=1, inplace=True)

df.dropna(inplace=True)

df.reset_index(drop=True, inplace=True)

df.head()
df_graph = df.loc[0:100]



plt.figure(figsize=(15,22))

plt.subplot(6,2,1)

plt.plot(df_graph['Hs'], color='blue')

plt.title('Significant Wave Height')



plt.subplot(6,2,2)

plt.plot(df_graph['Hmax'], color='red')

plt.title('Maximum Wave Height')



plt.subplot(6,2,3)

plt.plot(df_graph['Tz'], color='orange')

plt.title('Zero Upcrossing Wave Period')



plt.subplot(6,2,4)

plt.plot(df_graph['Tp'], color='brown')

plt.title('The Peak Energy Wave Period')



plt.subplot(6,2,5)

plt.plot(df_graph['Peak Direction'], color='purple')

plt.title('Direction Related to True North')



plt.subplot(6,2,6)

plt.plot(df_graph['SST'], color='green')

plt.title('Sea Surface Temperature')

plt.show();
print(df.info())
df.describe()
plt.figure(figsize=(7,7))

sns.heatmap(df.corr(), linewidth=.1, annot=True, cmap='YlGnBu')

plt.title('Correlation Matrix')

plt.show();
from sklearn.preprocessing import MinMaxScaler



# Scaling all the values between 0 and 1

scaler = MinMaxScaler(feature_range=(0,1))

data = scaler.fit_transform(df)

print('Shape of the scaled data matrix: ', data.shape)
# Separete data into 2 groups for train and test

train = data[:42000,]

test = data[42000: ,]



# Shapes of our datasets

print('Shape of train data: ', train.shape)

print('Shape of test data: ', test.shape)
# Separete every 30 samples as the input and get the 31st sample as the output.

def prepare_data(data):

    databatch = 30

    x_list = []

    y_list = []

    

    for i in range(len(data)-databatch-1):

        x_list.append(data[i:i+databatch])

        y_list.append(data[i+databatch+1])

        

    X_data = np.array(x_list)

    X_data = np.reshape(X_data, (X_data.shape[0], X_data.shape[2], X_data.shape[1]))

    y_data = np.array(y_list)

    

    return X_data, y_data
# Executing the separation

X_train, y_train = prepare_data(train)

X_test, y_test = prepare_data(test)

print('X_train Shape : ', X_train.shape, 'y_train shape :', y_train.shape)

print('X_test Shape  : ', X_test.shape, ' y_test shape  :', y_test.shape)
from keras.models import Sequential

from keras.layers import Dense, Dropout, LSTM

from keras.optimizers import Adam

from keras.callbacks import ModelCheckpoint



def lstm_model(x_data, y_data, num_epochs, batch_size, learning_rate):

    # Creating the model

    model = Sequential()

    # Adding the first layer

    model.add(LSTM(32, input_shape=(x_data.shape[1], x_data.shape[2]), return_sequences=True))

    # Adding the second layer 

    model.add(LSTM(16, return_sequences=True))

    # Adding a dropout value in order to prevent overfiting

    model.add(Dropout(0.2))

    # Adding the third layer

    model.add(LSTM(10))

    # Adding the output layer. 6 nodes are selected because the data has 6 features

    model.add(Dense(6))

    

    # Choosing the optimizer

    optimizer = Adam(lr=learning_rate)

    

    # Compiling the model

    model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['accuracy'])

    

    # Fitting the model

    history = model.fit(x_data, y_data, validation_split=0.25, epochs=num_epochs, batch_size=batch_size)

    

    return model, history
history = lstm_model(X_train, y_train, num_epochs=15, batch_size=200, learning_rate=.001)
plt.figure(figsize=(15,5))

plt.subplot(1,2,1)

plt.plot(history[1].history['accuracy'], color='blue', label='Train accuracy')

plt.plot(history[1].history['val_accuracy'], color='red', label='Validation accuracy')

plt.title('Train vs Validation Accuracy')

plt.xlabel('Number of Epochs')

plt.legend()

plt.subplot(1,2,2)

plt.plot(history[1].history['loss'], color='blue', label='Train Loss')

plt.plot(history[1].history['val_loss'], color='red', label='Validation Loss')

plt.title('Train vs Validation Loss')

plt.xlabel('Number of Epochs')

plt.legend()

plt.show();
# Defining function to predict datas

def predicting(data, y_real):

    predicted_data = history[0].predict(data)

    # Invert scaling process to get the normal values range for the features 

    predicted_data = scaler.inverse_transform(predicted_data)

    y_real = scaler.inverse_transform(y_real)

    

    return predicted_data, y_real
# Executing predictions

train_prediction, y_train = predicting(X_train, y_train)

test_prediction, y_test = predicting(X_test, y_test)
# Defining function to investigate the root of mean squared errors (RMSE) between predicted and real data



import math

from sklearn.metrics import mean_squared_error



def examine_rmse(y_data, predicted_data):

    Score_Hs = math.sqrt(mean_squared_error(y_data[:,0], predicted_data[:,0]))

    Score_Hmax = math.sqrt(mean_squared_error(y_data[:,1], predicted_data[:,1]))

    Score_Tz = math.sqrt(mean_squared_error(y_data[:,2], predicted_data[:,2]))

    Score_Tp = math.sqrt(mean_squared_error(y_data[:,3], predicted_data[:,3]))

    Score_Dir = math.sqrt(mean_squared_error(y_data[:,4], predicted_data[:,4]))

    Score_SST = math.sqrt(mean_squared_error(y_data[:,5], predicted_data[:,5]))

    

    print('RMSE_Hs       : ', Score_Hs)

    print('RMSE_Hmax     : ', Score_Hmax)

    print('RMSE_Tz       : ', Score_Tz)

    print('RMSE_Tp       : ', Score_Tp)

    print('RMSE_Direction: ', Score_Dir)

    print('RMSE_SST      : ', Score_SST)
# Executing the RMSE comparison

print('Trainin Data Errors')

print(examine_rmse(y_train, train_prediction),'\n')

print('Test Data Errors')

print(examine_rmse(y_test, test_prediction))
plt.figure(figsize=(17,25))





plt.subplot(6,2,1)

plt.plot(test_prediction[1300:,0], color='red', alpha=0.7, label='prediction')

plt.plot(y_test[1300:,0], color='blue', alpha=0.5, label='real')

plt.title('Significant Wave Height')

plt.legend()

plt.grid(b=True, axis='y')



plt.subplot(6,2,2)

plt.plot(test_prediction[1300:,1], color='red', alpha=0.7, label='prediction')

plt.plot(y_test[1300:,1], color='blue', alpha=0.5, label='real')

plt.title('Maximum Wave Height')

plt.legend()

plt.grid(b=True, axis='y')



plt.subplot(6,2,3)

plt.plot(test_prediction[1300:,2], color='red', alpha=0.7, label='prediction')

plt.plot(y_test[1300:,2], color='blue', alpha=0.5, label='real')

plt.title('Zero Upcrossing Wave Period')

plt.legend()

plt.grid(b=True, axis='y')



plt.subplot(6,2,4)

plt.plot(test_prediction[1300:,3], color='red', alpha=0.7, label='prediction')

plt.plot(y_test[1300:,3], color='blue', alpha=0.5, label='real')

plt.title('Peak Energy Wave Period')

plt.legend()

plt.grid(b=True, axis='y')



plt.subplot(6,2,5)

plt.plot(test_prediction[1300:,4], color='red', alpha=0.7, label='prediction')

plt.plot(y_test[1300:,4], color='blue', alpha=0.5, label='real')

plt.title('Direction Related to True North')

plt.legend()

plt.grid(b=True, axis='y')



plt.subplot(6,2,6)

plt.plot(test_prediction[1300:,5], color='red', alpha=0.7, label='prediction')

plt.plot(y_test[1300:,5], color='blue', alpha=0.5, label='real')

plt.title('Sea Surface Temperature')

plt.legend()

plt.grid(b=True, axis='y')

plt.show();