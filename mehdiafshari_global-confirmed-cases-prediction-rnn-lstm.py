import requests
import numpy as np
import pandas as pd
import io
import tensorflow as tf
import matplotlib.pyplot as plt
from random import random
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
BASE_URL = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/'
CONFIRMED = 'time_series_covid19_confirmed_global.csv'
DEATH = 'time_series_covid19_deaths_global.csv'
RECOVERED = 'time_series_covid19_recovered_global.csv'
CONFIRMED_US = 'time_series_covid19_confirmed_US.csv'
DEATH_US = 'time_series_covid19_deaths_US.csv'

def get_covid_data(subset = 'CONFIRMED'):
    """This function returns the latest available data subset of COVID-19. 
        The returned value is in pandas DataFrame type.
    Args:
        subset (:obj:`str`, optional): Any value out of 5 subsets of 'CONFIRMED',
        'DEATH', 'RECOVERED', 'CONFIRMED_US' and 'DEATH_US' is a valid input. If the value
        is not chosen or typed wrongly, CONFIRMED subet will be returned.
    """    
    switcher =  {
                'CONFIRMED'     : BASE_URL + CONFIRMED,
                'DEATH'         : BASE_URL + DEATH,
                'RECOVERED'     : BASE_URL + RECOVERED,
                'CONFIRMED_US'  : BASE_URL + CONFIRMED_US,
                'DEATH_US'      : BASE_URL + DEATH_US,
                }

    CSV_URL = switcher.get(subset, BASE_URL + CONFIRMED)

    with requests.Session() as s:
        download        = s.get(CSV_URL)
        decoded_content = download.content.decode('utf-8')
        data            = pd.read_csv(io.StringIO(decoded_content))

    return data
def covid_model(input_len):
  covid_19_RNN=tf.keras.models.Sequential()
  #input layer
  covid_19_RNN.add(tf.keras.layers.LSTM(units = 250, return_sequences=True,input_shape=(input_len,1)))
  covid_19_RNN.add(tf.keras.layers.Dropout(0.01))
  #1st hidden layer
  covid_19_RNN.add(tf.keras.layers.LSTM(units = 150, return_sequences=True))
  covid_19_RNN.add(tf.keras.layers.Dropout(0.01))
  # #2nd hidden layer
  covid_19_RNN.add(tf.keras.layers.LSTM(units = 200, return_sequences=True))
  covid_19_RNN.add(tf.keras.layers.Dropout(0.02))
  #3rd hidden layer
  covid_19_RNN.add(tf.keras.layers.LSTM(units = 150, return_sequences=True))
  covid_19_RNN.add(tf.keras.layers.Dropout(0.02))
  #4th hidden layer
  covid_19_RNN.add(tf.keras.layers.LSTM(units = 125, return_sequences=True))
  covid_19_RNN.add(tf.keras.layers.Dropout(0.04))
  #output layer
  covid_19_RNN.add(tf.keras.layers.Dense(units = 1))


  #compile the model
  covid_19_RNN.compile(optimizer='adam',loss='mae')
  return covid_19_RNN
# Initializing Parameters

TEST_PORTION = 0
TEST_SIZE = 14 
OBSERVATION_WINDOW_SIZE = 14
AUGMENTATION_LEVEL = 50
AUGMENTATION_SCORE = 0.1
INPUT_FILE = 'CONFIRMED'
"""
INPUT_FILE : 'CONFIRMED', 'DEATH', 'RECOVERED', 'CONFIRMED_US' and 'DEATH_US'
"""
# GPU settings initialization

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)

device_name = tf.test.gpu_device_name()
if not device_name == '/device:GPU:0':
  raise SystemError('GPU device not found')
print('Found GPU at: {}'.format(device_name))
# Reading the Input File

df = get_covid_data(subset = INPUT_FILE)
print(df)

if 'US' in INPUT_FILE:
  data_start = 10
else:
  data_start = 3

confirmed_data=pd.DataFrame(df.sum(axis=0))[data_start:]
dates=list(confirmed_data.index)
test_size = np.asarray(max( np.round( len(dates) * TEST_PORTION ) , TEST_SIZE )).astype(int)
time_steps = max( OBSERVATION_WINDOW_SIZE , 1)
aug_steps = max( AUGMENTATION_LEVEL , 1)
scaler_threshold = min( AUGMENTATION_SCORE, 0.2 )

day=[]
for i in range(0,len(dates)):
  day.append(i+1)
day=np.array(day)
# Augmentation and Preprocessing of the Train Data

scaler = MinMaxScaler( feature_range=(random()*0.2,1-random()*0.2) )
train_data = confirmed_data[:-test_size]

x_data=[]
y_data=[]

for j in range(0,aug_steps):
    for i in range(time_steps,len(train_data)):
        scaler=MinMaxScaler( feature_range=(random()*scaler_threshold, 1 - random()*scaler_threshold) )
        augmentation = scaler.fit_transform(train_data[i-time_steps:i+1])
        x_data.append(augmentation[:time_steps])
        y_data.append(augmentation[time_steps])

x_data = np.array(x_data)
x_data = np.reshape(x_data,(x_data.shape[0],x_data.shape[1],1))
y_data = np.array(y_data)
# Fitting the model to the augmented data

covid_19_RNN = covid_model(x_data.shape[1])
covid_19_RNN.fit(x_data ,y_data , batch_size=2048 , epochs=200 , validation_split=0.4 , verbose=2)
# The test data sequence generation and preprocessing

test_data = confirmed_data.values
test_data = test_data
test_data = np.array(test_data).reshape(-1,1)
# Prediction using the trained covid_19_RNN model

scaler=MinMaxScaler( feature_range = (scaler_threshold, 1 - scaler_threshold) )
prediction_result = []

for i in range(time_steps ,len(test_data)):
  x_test = []
  x_test.append( scaler.fit_transform(test_data[ i - time_steps : i ]) )
  x_test = np.reshape(x_test[0],(1 , x_test[0].shape[0], 1))
  predicted_covid_19_spread = covid_19_RNN.predict(x_test)[0][-1]
  predict_val = scaler.inverse_transform(predicted_covid_19_spread.reshape(1,1))
  predict_val = predict_val.astype(int)
  prediction_result.append(predict_val[0][0])

y_test = test_data[time_steps : len(test_data)]
y_test = y_test.reshape(len(y_test)).astype(int)
# Test Data Results

result_error = prediction_result - y_test
result_error_percentage = 100*np.divide(result_error, y_test)

test_dates  = np.asarray(dates[-len(y_test):]).reshape(len(y_test))
real_value  = y_test.reshape(len(y_test)).astype(int)
predition   = np.asarray(prediction_result)
error_p     = result_error_percentage.reshape(len(y_test))
error_p     = np.around(error_p, decimals=2)

Results = {'Date': test_dates, 'Real Value [No.]': real_value, 'Predicted Value [No.]': predition, 'Error Percentage [%]':error_p}
df_results = pd.DataFrame(data=Results)
print(df_results)
MAEP  = mean_absolute_error(predition, y_test)
MAEP  = np.around(MAEP, decimals=0)
print('Mean Absolute Error of Prediction                 :' + str(MAEP))
MAPEP = mean_absolute_error(error_p,np.zeros(len(error_p)))
MAPEP = np.around(MAPEP, decimals=2)
print('Mean Absolute of Prediction Error Percentages  [%]:' + str(MAPEP))
plt.figure(2, figsize=(12,9))
plt.plot(range(0,len(confirmed_data)),confirmed_data[0],label="True Number of corona-cases", c='r', marker='.')
plt.plot(range(len(confirmed_data) - len(prediction_result) ,len(confirmed_data)),prediction_result,label="Prediction of Number of corona-cases",c='b', marker='.')
plt.xlabel('Day Number')
plt.ylabel('Number of people')
plt.legend()
plt.grid(which='both', axis='both')
plt.title('Day by Day Prediction of Time Series COVID-19 Cases - ' + INPUT_FILE.upper() + ' GLOBAL')
plt.show()
# The test data sequence generation and preprocessing

test_data = confirmed_data.values
test_data = test_data[-(test_size + time_steps):]
test_data = np.array(test_data).reshape(-1,1)
prediction = np.append(test_data[-14:], 0*test_data[:14])
prediction = np.asarray(prediction).reshape(28,1)

scaler=MinMaxScaler( feature_range = (scaler_threshold, 1 - scaler_threshold) )
prediction_result = []

for i in range(time_steps ,len(prediction)):
  x_test = []
  x_test.append( scaler.fit_transform(prediction[ i - time_steps : i ]) )
  x_test = np.reshape(x_test[0],(1 , x_test[0].shape[0], 1))
  predicted_covid_19_spread = covid_19_RNN.predict(x_test)[0][-1]
  predict_val = scaler.inverse_transform(predicted_covid_19_spread.reshape(1,1))
  predict_val = predict_val.astype(int)
  prediction[i]= predict_val
  prediction_result.append(predict_val[0][0])

plt.figure(2, figsize=(12,9))
plt.plot(range(0,len(confirmed_data)),confirmed_data[0],label="True Number of corona-cases", c='r', marker='.')
plt.plot(range(len(confirmed_data) - test_size + 14 ,len(confirmed_data) + 14),prediction_result,label="Prediction of Number of corona-cases",c='b', marker='.')
plt.xlabel('Day Number')
plt.ylabel('Number of people')
plt.legend()
plt.grid(which='both', axis='both')
plt.title('14 Days Prediction of Time Series COVID-19 Cases - ' + INPUT_FILE.upper() + ' GLOBAL')
plt.show()