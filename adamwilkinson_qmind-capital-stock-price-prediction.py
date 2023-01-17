# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import matplotlib.pyplot as plt
import statsmodels.tsa.seasonal as smt
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import random
import datetime as dt
from sklearn import linear_model 
from sklearn.metrics import mean_squared_error
import plotly

# import the relevant Keras modules
from keras.models import Sequential
from keras.layers import Activation, Dense
from keras.layers import LSTM
from keras.layers import Dropout

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))
import os
os.chdir('../input')

#read data
# kernels let us navigate through the zipfile as if it were a directory

# trying to read a file of size zero will throw an error, so skip them
# filenames = [x for x in os.listdir() if x.endswith('.txt') and os.path.getsize(x) > 0]
# filenames = random.sample(filenames,1)
filename = 'all_stocks_5yr.csv'
print(filename)

dataI = pd.read_csv(filename)
flixdata = dataI.loc[dataI['Name'] == 'NFLX']
flixdata = flixdata.loc[:,['date','close']]
flixdata['date'] = pd.to_datetime(flixdata['date'])

data = [] 
data.append(flixdata)


flixdata2 = flixdata

flixplot = flixdata2.plot(x='date',y='close',title='Netflix Closing Price vs Date')
flixplot.set_xlabel('Date')
flixplot.set_ylabel('Closing Price ($)')

flixcloseprice = flixdata2['close']
flixdata_norm = (flixcloseprice - flixcloseprice.mean())/(flixcloseprice.max() - flixcloseprice.min())

flixdata2['close norm'] = flixdata_norm
#flixdata = pd.concat([flixdata['date'],flixdata_norm], axis=0, ignore_index=True)
#flixdata = flixdata['date'].join(flixdata_norm)

flixplot2 = flixdata.plot(x='date',y='close norm',title='Netflix Closing Price Norm vs Date')
df = data[0]
window_len = 10

#Create a data point (i.e. a date) which splits the training and testing set
split_date = list(data[0]["date"][-(10*window_len+1):])[0]

#Split the training and test set
training_set, test_set = df[df['date'] < split_date], df[df['date'] >= split_date]
training_set = training_set.drop(['date'], 1)
test_set = test_set.drop(['date'], 1)

#Create windows for training
LSTM_training_inputs = []
for i in range(len(training_set)-window_len):
    temp_set = training_set[i:(i+window_len)].copy()
        
    for col in list(temp_set):
        temp_set[col] = temp_set[col]/temp_set[col].iloc[0] - 1
    
    LSTM_training_inputs.append(temp_set)
LSTM_training_outputs = (training_set['close'][window_len:].values/training_set['close'][:-window_len].values)-1

LSTM_training_inputs = [np.array(LSTM_training_input) for LSTM_training_input in LSTM_training_inputs]
LSTM_training_inputs = np.array(LSTM_training_inputs)

#Create windows for testing
LSTM_test_inputs = []
for i in range(len(test_set)-window_len):
    temp_set = test_set[i:(i+window_len)].copy()
    
    for col in list(temp_set):
        temp_set[col] = temp_set[col]/temp_set[col].iloc[0] - 1
    
    LSTM_test_inputs.append(temp_set)
LSTM_test_outputs = (test_set['close'][window_len:].values/test_set['close'][:-window_len].values)-1

LSTM_test_inputs = [np.array(LSTM_test_inputs) for LSTM_test_inputs in LSTM_test_inputs]
LSTM_test_inputs = np.array(LSTM_test_inputs)

def build_model(inputs, output_size, neurons, activ_func="linear",
                dropout=0.10, loss="mse", optimizer="adam"):
    
    model = Sequential()

    model.add(LSTM(neurons, input_shape=(inputs.shape[1], inputs.shape[2])))
    model.add(Dropout(dropout))
    model.add(Dense(units=output_size))
    model.add(Activation(activ_func))

    model.compile(loss=loss, optimizer=optimizer)
    return model

# initialise model architecture
nn_model = build_model(LSTM_training_inputs, output_size=1, neurons = 32)
# model output is next price normalised to 5th previous closing price
# train model on data
# note: eth_history contains information on the training error per epoch
nn_history = nn_model.fit(LSTM_training_inputs, LSTM_training_outputs, 
                            epochs=5, batch_size=1, verbose=2, shuffle=False)

plt.plot(LSTM_test_outputs, label = "actual")
plt.plot(nn_model.predict(LSTM_test_inputs), label = "predicted")
plt.legend()
plt.show()
MSE = mean_squared_error(LSTM_test_outputs, nn_model.predict(LSTM_test_inputs))
print('The Mean Squared Error is: {}'.format(MSE))
