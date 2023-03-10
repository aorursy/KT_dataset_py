import numpy as np

import matplotlib.pyplot as plt

import matplotlib.pyplot as plt2

import pandas as pd

from pandas import datetime

import math, time

import itertools

from sklearn import preprocessing

import datetime

from sklearn.metrics import mean_squared_error

from math import sqrt

from keras.models import Sequential

from keras.layers.core import Dense, Dropout, Activation

from keras.layers.recurrent import LSTM

from keras.models import load_model

import keras

import pandas_datareader.data as web

import h5py
stock_name = 'INFY'

seq_len = 22

rate = 0.2

shape = [4, seq_len, 1] # feature, window, output

neurons = [128, 128, 32, 1]

epochs = 300
def get_stock_data(stock_name, normalize=True):

    start = datetime.datetime(2013, 1, 1)

    end = datetime.datetime(2018,12,31)

    df = web.DataReader(stock_name, "yahoo", start, end)

    df.drop(['Volume', 'Adj Close'], 1, inplace=True)

    

    if normalize:        

        min_max_scaler = preprocessing.MinMaxScaler()

        df['Open'] = min_max_scaler.fit_transform(df.Open.values.reshape(-1,1))

        df['High'] = min_max_scaler.fit_transform(df.High.values.reshape(-1,1))

        df['Low'] = min_max_scaler.fit_transform(df.Low.values.reshape(-1,1))

        df['Close'] = min_max_scaler.fit_transform(df['Close'].values.reshape(-1,1))

    return df
start = datetime.datetime(2013, 1, 1)

end = datetime.datetime(2018,12,31)

df = web.DataReader(stock_name, "yahoo", start, end)

df.drop(['Volume', 'Adj Close'], 1, inplace=True)

df.head()
df = get_stock_data(stock_name, normalize=True)
df.head()
df.tail()
def plot_stock(stock_name):

    df = get_stock_data(stock_name, normalize=True)

    print(df.head())

    plt.plot(df['Close'], color='red', label='Close')

    plt.legend(loc='best')

    plt.show()
plot_stock(stock_name)
def load_data(stock, seq_len):

    amount_of_features = len(stock.columns)

    data = stock.as_matrix() 

    sequence_length = seq_len + 1 # index starting from 0

    result = []

    

    for index in range(len(data) - sequence_length): # maxmimum date = lastest date - sequence length

        result.append(data[index: index + sequence_length]) # index : index + 22days

    

    result = np.array(result)

    row = round(0.9 * result.shape[0]) # 90% split

    

    train = result[:int(row), :] # 90% date

    X_train = train[:, :-1] # all data until day m

    y_train = train[:, -1][:,-1] # day m + 1 adjusted close price

    

    X_test = result[int(row):, :-1]

    y_test = result[int(row):, -1][:,-1] 



    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], amount_of_features))

    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], amount_of_features))  



    return [X_train, y_train, X_test, y_test]
'''from sklearn import model_selection

from sklearn.model_selection import train_test_split

from sklearn.model_selection import TimeSeriesSplit



def load_data_1(stock, split_point):

    amount_of_features = len(stock.columns)

    stock = np.array(stock)

    X = stock[:,:-1]

    y = stock[:,-1]

    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size = split_point, shuffle = False)

    

    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], amount_of_features))

    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], amount_of_features))  

    

    return [X_train, y_train, X_test, y_test]  '''
X_train, y_train, X_test, y_test = load_data(df, seq_len)
# X_train, y_train, X_test, y_test = load_data_1(df, .25)
X_train.shape, X_test.shape, y_train.shape, y_test.shape
X_train.itemsize
# X_test.tail()




X_train.shape[0], X_train.shape[1], X_train.shape[2]
# y_train.head()
def build_model2(layers, neurons, d):

    model = Sequential()

    

    model.add(LSTM(neurons[0], input_shape=(layers[1], layers[0]), return_sequences=True))

    model.add(Dropout(rate, noise_shape=None, seed=None))

        

    model.add(LSTM(neurons[1], input_shape=(layers[1], layers[0]), return_sequences=False))

    model.add(Dropout(rate, noise_shape=None, seed=None))

        

    model.add(Dense(neurons[2],kernel_initializer="uniform",activation='relu'))        

    model.add(Dense(neurons[3],kernel_initializer="uniform",activation='linear'))

    # model = load_model('my_LSTM_stock_model1000.h5')

    # adam = keras.optimizers.Adam(decay=0.2)

    model.compile(loss='mse',optimizer='adam', metrics=['accuracy'])

    model.summary()

    return model
model = build_model2(shape, neurons, rate)

# layers = [4, 22, 1]
model.summary()
model.fit(

    X_train,

    y_train,

    batch_size=512,

    epochs=epochs,

    validation_split=0.1,

    verbose=1)
def model_score(model, X_train, y_train, X_test, y_test):

    trainScore = model.evaluate(X_train, y_train, verbose=0)

    print('Train Score: %.5f MSE (%.2f RMSE)' % (trainScore[0], math.sqrt(trainScore[0])))



    testScore = model.evaluate(X_test, y_test, verbose=0)

    print('Test Score: %.5f MSE (%.2f RMSE)' % (testScore[0], math.sqrt(testScore[0])))

    return trainScore[0], testScore[0]
model_score(model, X_train, y_train, X_test, y_test)
def percentage_difference(model, X_test, y_test):

    percentage_diff=[]



    p = model.predict(X_test)

    for u in range(len(y_test)): # for each data index in test data

        pr = p[u][0] # pr = prediction on day u



        percentage_diff.append((pr-y_test[u]/pr)*100)

    return p
p = percentage_difference(model, X_test, y_test)
len(y_test)
def denormalize(stock_name, normalized_value):

    start = datetime.datetime(2000, 1, 1)

    end = datetime.date.today()

    df = web.DataReader(stock_name, "yahoo", start, end)

    

    df = df['Close'].values.reshape(-1,1)

    normalized_value = normalized_value.reshape(-1,1)

    

    #return df.shape, p.shape

    min_max_scaler = preprocessing.MinMaxScaler()

    a = min_max_scaler.fit_transform(df)

    new = min_max_scaler.inverse_transform(normalized_value)

    return new
def plot_result(stock_name, normalized_value_p, normalized_value_y_test):

    newp = denormalize(stock_name, normalized_value_p)

    newy_test = denormalize(stock_name, normalized_value_y_test)

    plt2.plot(newp, color='red', label='Prediction')

    plt2.plot(newy_test,color='blue', label='Actual')

    plt2.legend(loc='best')

    plt2.title('The test result for {}'.format(stock_name))

    plt2.xlabel('Days')

    plt2.ylabel('Close')

    plt2.show()

    return newp,newy_test
newp,newy_test = plot_result(stock_name, p, y_test)


my_pred = pd.DataFrame(columns=['pred', 'actual'])

my_pred.head()
my_pred['pred'] = newp.tolist()

my_pred['actual'] = newy_test.tolist()
my_pred.head()
my_pred.to_csv('LSTM_Stock_prediction_1.csv', sep=',', encoding='utf-8')
model.save('LSTM_Stock_prediction-1.h5')
stock_name = 'INFY'

seq_len = 22

shape = [4, seq_len, 1] # feature, window, output

neurons = [128, 128, 32, 1]

epochs = 300
def quick_measure(stock_name, seq_len, rate, shape, neurons, epochs):

    df = get_stock_data(stock_name)

    X_train, y_train, X_test, y_test = load_data(df, seq_len)

    model = build_model2(shape, neurons, rate)

    model.fit(X_train, y_train, batch_size=512, epochs=epochs, validation_split=0.1, verbose=1)

    # model.save('LSTM_Stock_prediction-20170429.h5')

    trainScore, testScore = model_score(model, X_train, y_train, X_test, y_test)

    return trainScore, testScore
dlist = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

neurons_LSTM = [32, 64, 128, 256, 512, 1024, 2048]

dropout_result = {}



for d in dlist:    

    trainScore, testScore = quick_measure(stock_name, seq_len, rate, shape, neurons, epochs)

    dropout_result[rate] = testScore
min_val = min(dropout_result.values())

min_val_key = [k for k, v in dropout_result.items() if v == min_val]

print (dropout_result)

print (min_val_key)
lists = sorted(dropout_result.items())

x,y = zip(*lists)

plt.plot(x,y)

plt.title('Finding the best hyperparameter')

plt.xlabel('Dropout')

plt.ylabel('Mean Square Error')

plt.show()
stock_name = 'INFY'

seq_len = 22

shape = [4, seq_len, 1] # feature, window, output

neurons = [128, 128, 32, 1]

epochslist = [10,20,30,40,50,60,70,80,90,100]
epochs_result = {}



for epochs in epochslist:    

    trainScore, testScore = quick_measure(stock_name, seq_len, rate, shape, neurons, epochs)

    epochs_result[epochs] = testScore
lists = sorted(epochs_result.items())

x,y = zip(*lists)

plt.plot(x,y)

plt.title('Finding the best hyperparameter')

plt.xlabel('Epochs')

plt.ylabel('Mean Square Error')

plt.show()
stock_name = 'INFY'

seq_len = 22

shape = [4, seq_len, 1] # feature, window, output

epochs = 90

dropout = 0.3

neuronlist1 = [32, 64, 128, 256, 512]

neuronlist2 = [16, 32, 64]

neurons_result = {}



for neuron_lstm in neuronlist1:

    neurons = [neuron_lstm, neuron_lstm]

    for activation in neuronlist2:

        neurons.append(activation)

        neurons.append(1)

        trainScore, testScore = quick_measure(stock_name, seq_len, rate, shape, neurons, epochs)

        neurons_result[str(neurons)] = testScore

        neurons = neurons[:2]    
lists = sorted(neurons_result.items())

x,y = zip(*lists)



plt.title('Finding the best hyperparameter')

plt.xlabel('neurons')

plt.ylabel('Mean Square Error')



plt.bar(range(len(lists)), y, align='center')

plt.xticks(range(len(lists)), x)

plt.xticks(rotation=90)



plt.show()
stock_name = 'INFY'

seq_len = 22

shape = [4, seq_len, 1] # feature, window, output

neurons = [512, 512, 64, 1]

epochs = 90

decaylist = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
def build_model3(layers, neurons, rate, decay):

    model = Sequential()

    

    model.add(LSTM(neurons[0], input_shape=(layers[1], layers[0]), return_sequences=True))

    model.add(Dropout(rate))

        

    model.add(LSTM(neurons[1], input_shape=(layers[1], layers[0]), return_sequences=False))

    model.add(Dropout(rate))

        

    model.add(Dense(neurons[2],kernel_initializer="uniform",activation='relu'))        

    model.add(Dense(neurons[3],kernel_initializer="uniform",activation='linear'))

    # model = load_model('my_LSTM_stock_model1000.h5')

    adam = keras.optimizers.Adam(decay=decay)

    model.compile(loss='mse',optimizer='adam', metrics=['accuracy'])

    model.summary()

    return model
def quick_measure(stock_name, seq_len, d, shape, neurons, epochs, decay):

    df = get_stock_data(stock_name)

    X_train, y_train, X_test, y_test = load_data(df, seq_len)

    model = build_model3(shape, neurons, rate, decay)

    model.fit(X_train, y_train, batch_size=512, epochs=epochs, validation_split=0.1, verbose=1)

    # model.save('LSTM_Stock_prediction-20170429.h5')

    trainScore, testScore = model_score(model, X_train, y_train, X_test, y_test)

    return trainScore, testScore
decay_result = {}



for decay in decaylist:    

    trainScore, testScore = quick_measure(stock_name, seq_len, rate, shape, neurons, epochs, decay)

    decay_result[decay] = testScore
lists = sorted(decay_result.items())

x,y = zip(*lists)

plt.plot(x,y)

plt.title('Finding the best hyperparameter')

plt.xlabel('Decay')

plt.ylabel('Mean Square Error')

plt.show()
stock_name = 'INFY'

neurons = [512, 512, 64, 1]

epochs = 90

rate = 0.3 #dropout

decay = 0.7
seq_len_list = [5, 10, 22, 60, 120, 180]



seq_len_result = {}



for seq_len in seq_len_list:

    shape = [4, seq_len, 1]

    

    trainScore, testScore = quick_measure(stock_name, seq_len, rate, shape, neurons, epochs, decay)

    seq_len_result[seq_len] = testScore
lists = sorted(seq_len_result.items())

x,y = zip(*lists)

plt.plot(x,y)

plt.title('Finding the best hyperparameter')

plt.xlabel('Days')

plt.ylabel('Mean Square Error')

plt.show()
seq_len = 60

shape = [4, seq_len, 1]

    

trainScore, testScore = quick_measure(stock_name, seq_len, rate, shape, neurons, epochs, decay)

seq_len_result[seq_len] = testScore
p = percentage_difference(model, X_test, y_test)
newp,newy_test = plot_result(stock_name, p, y_test)
my_pred = pd.DataFrame(columns=['pred', 'actual'])

my_pred.head()



my_pred['pred'] = newp.tolist()

my_pred['actual'] = newy_test.tolist()



my_pred.head()



my_pred.to_csv('LSTM_Stock_prediction_Close.csv', sep=',', encoding ='utf-8')



model.save('LSTM_Stock_prediction_Close.h5')
y_test.shape, y_train.shape
my_pred.head()
#my_pred_1 = pd.DataFrame(columns=['pred', 'actual'])

#my_pred_1.head()



#my_pred_1['pred'] = newp

#my_pred_1['actual'] = newy_test



#my_pred.head()

print(newp, newy_test)
my_pred_data = np.concatenate((newy_test, newp), axis=1)

print(my_pred_data.view)
from statsmodels import robust

along_row = robust.mad(my_pred_data, axis=1)

along_column = robust.mad(my_pred_data, axis=0)
print(along_column)
print(along_row)
np.savetxt("LSTM_col_y-test_pred.csv", my_pred_data, delimiter=",")