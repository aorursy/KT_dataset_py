import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import matplotlib.dates as mdates

from keras.models import Sequential

from keras.layers import Dense, Dropout,GRU

from keras import optimizers 

from sklearn.preprocessing import MinMaxScaler



seed = 1234

np.random.seed(seed)

plt.style.use('ggplot')
dataraw = pd.read_csv('../input/bitcoin-data-from-2014-to-2020/BTC-USD.csv',index_col='Date', parse_dates=['Date'])

dataraw
# use feature 'Date' & 'Close'

dataset = pd.DataFrame(dataraw['Close'])

print(dataset.describe())

print(' Count row of data: ',len(dataset))
#Min-Max Normalization

dataset_norm = dataset.copy()

dataset[['Close']]

scaler = MinMaxScaler()

dataset_norm['Close'] = scaler.fit_transform(dataset[['Close']])

dataset_norm
fig = plt.figure(figsize=(10, 4))

plt.plot(dataset_norm)

plt.xlabel('Date')

plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

plt.title('Data Normalized')

plt.show()
# Partition data into data train, val & test

totaldata = dataset.values

totaldatatrain = int(len(totaldata)*0.7)

totaldataval = int(len(totaldata)*0.1)

totaldatatest = int(len(totaldata)*0.2)



# Store data into each partition

training_set = dataset_norm[0:totaldatatrain]

val_set=dataset_norm[totaldatatrain:totaldatatrain+totaldataval]

test_set = dataset_norm[totaldatatrain+totaldataval:]
# graph data training

fig = plt.figure(figsize=(10, 4))

plt.plot(training_set)

plt.xlabel('Date')

plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

plt.title('Data Training')

plt.show()
# graph data validation

fig = plt.figure(figsize=(10, 4))

plt.plot(val_set)

plt.xlabel('Date')

plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

plt.title('Data Validation')

plt.show()
# graph data test

fig = plt.figure(figsize=(10, 4))

plt.plot(test_set)

plt.xlabel('Date')

plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

plt.title('Data Test')

plt.show()
# Initiaton value of lag

lag = 2

# Sliding windows function

def create_sliding_windows(data,len_data,lag):

    x=[]

    y=[]

    for i in range(lag,len_data):

        x.append(data[i-lag:i,0])

        y.append(data[i,0]) 

    return np.array(x),np.array(y)



# Formating data into array for create sliding windows

array_training_set = np.array(training_set)

array_val_set = np.array(val_set)

array_test_set = np.array(test_set)



# Create sliding windows into training data

x_train, y_train = create_sliding_windows(array_training_set,len(array_training_set), lag)

x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))

# Create sliding windows into validation data

x_val,y_val = create_sliding_windows(array_val_set,len(array_val_set),lag)

x_val = np.reshape(x_val, (x_val.shape[0],x_val.shape[1],1))

# Create sliding windows into test data

x_test,y_test = create_sliding_windows(array_test_set,len(array_test_set),lag)

x_test = np.reshape(x_test, (x_test.shape[0],x_test.shape[1],1))
# Hyperparameters

learning_rate = 0.0001

hidden_unit = 64

batch_size=256

epoch = 100



# Architecture Gated Recurrent Unit

regressorGRU = Sequential()



# First GRU layer with dropout

regressorGRU.add(GRU(units=hidden_unit, return_sequences=True, input_shape=(x_train.shape[1],1), activation = 'tanh'))

regressorGRU.add(Dropout(0.2))

# Second GRU layer with dropout

regressorGRU.add(GRU(units=hidden_unit, return_sequences=True, activation = 'tanh'))

regressorGRU.add(Dropout(0.2))

# Third GRU layer with dropout

regressorGRU.add(GRU(units=hidden_unit, return_sequences=False, activation = 'tanh'))

regressorGRU.add(Dropout(0.2))



# Output layer

regressorGRU.add(Dense(units=1))



# Compiling the Gated Recurrent Unit

regressorGRU.compile(optimizer=optimizers.Adam(lr=learning_rate),loss='mean_squared_error')



# Fitting ke data training dan data validation

pred = regressorGRU.fit(x_train, y_train, validation_data=(x_val,y_val), batch_size=batch_size, epochs=epoch)
# Graph model loss (train loss & val loss)

fig = plt.figure(figsize=(10, 4))

plt.plot(pred.history['loss'], label='train loss')

plt.plot(pred.history['val_loss'], label='val loss')

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(loc='upper right')

plt.show()
# Tabel value of training loss & validation loss

learningrate_parameter = learning_rate

train_loss=pred.history['loss'][-1]

validation_loss=pred.history['val_loss'][-1]

learningrate_parameter=pd.DataFrame(data=[[learningrate_parameter, train_loss, validation_loss]],

                                    columns=['Learning Rate', 'Training Loss', 'Validation Loss'])

learningrate_parameter.set_index('Learning Rate')
# Implementation model into data test

y_pred_test = regressorGRU.predict(x_test)



# Invert normalization min-max

y_pred_invert_norm = scaler.inverse_transform(y_pred_test)

# Comparison data test with data prediction

datacompare = pd.DataFrame()

datatest=np.array(dataset['Close'][totaldatatrain+totaldataval+lag:])

datapred= y_pred_invert_norm



datacompare['Data Test'] = datatest

datacompare['Prediction Results'] = datapred

datacompare
# Calculatre value of Root Mean Square Error 

def rmse(datatest, datapred):

    return np.round(np.sqrt(np.mean((datapred - datatest) ** 2)), 4)

print('Root Mean Square Error :',rmse(datatest, datapred))



def mape(datatest, datapred): 

    return np.round(np.mean(np.abs((datatest - datapred) / datatest) * 100), 4)

    

print('Mean Absolute Percentage Error : ', mape(datatest, datapred), '%')
# Create graph data test and prediction result

plt.figure(num=None, figsize=(10, 4), dpi=80,facecolor='w', edgecolor='k')

plt.title('Graph Comparison Data Actual and Data Prediction')

plt.plot(datacompare['Data Test'], color='red',label='Data Test')

plt.plot(datacompare['Prediction Results'], color='blue',label='Prediction Results')

plt.xlabel('Day')

plt.ylabel('Price')

plt.legend()

plt.show()
