import numpy as np # linear algebra

from numpy import newaxis

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import os

from keras.layers.core import Dense, Activation, Dropout

from keras.layers.recurrent import LSTM, GRU

from keras.models import Sequential

from keras import optimizers



print(os.listdir("../input"))
def normalise_windows(window_data):

    # A support function to normalize a dataset

    normalised_data = []

    for window in window_data:

        normalised_window = [((float(p) / float(window[0])) - 1) for p in window]

        normalised_data.append(normalised_window)

    return normalised_data



def load_data(datasetname, column, seq_len, normalise_window):

    # A support function to help prepare datasets for an RNN/LSTM/GRU

    data = datasetname.loc[:,column]

    sequence_length = seq_len + 1

    result = []

    

    for index in range(len(data) - sequence_length):

        result.append(data[index: index + sequence_length])

    

    if normalise_window:

        #result = sc.fit_transform(result)

        result = normalise_windows(result)

    result = np.array(result)



    #Last 10% is used for validation test, first 90% for training

    row = round(0.9 * result.shape[0])

    train = result[:int(row), :]

    np.random.shuffle(train)

    x_train = train[:, :-1]

    y_train = train[:, -1]

    x_test = result[int(row):, :-1]

    y_test = result[int(row):, -1]

    

    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))  



    return [x_train, y_train, x_test, y_test]



def predict_sequence_full(model, data, window_size):

    #Shift the window by 1 new prediction each time, re-run predictions on new window

    curr_frame = data[0]

    predicted = []

    for i in range(len(data)):

        predicted.append(model.predict(curr_frame[newaxis,:,:])[0,0])

        curr_frame = curr_frame[1:]

        curr_frame = np.insert(curr_frame, [window_size-1], predicted[-1], axis=0)

    return predicted



def plot_results(predicted_data, true_data): 

    fig = plt.figure(facecolor='white') 

    ax = fig.add_subplot(111) 

    ax.plot(true_data, label='True Data') 

    plt.plot(predicted_data, label='Prediction') 

    plt.legend() 

    plt.show() 
# Let's get the stock data

dataset = pd.read_csv('../input/IBM_2006-01-01_to_2018-01-01.csv', index_col='Date', parse_dates=['Date'])

dataset.head()
# Prepare the dataset, note that the stock price data will be normalized between 0 and 1

# A label is the thing we're predicting

# A feature is an input variable, in this case a stock price

# Selected 'Close' (stock pric at closing) attribute for prices. Let's see what it looks like

Enrol_window = 100

feature_train, label_train, feature_test, label_test = load_data(dataset, 'Close', Enrol_window, True)



dataset["Close"][:'2016'].plot(figsize=(16,4),legend=True)

dataset["Close"]['2017':].plot(figsize=(16,4),legend=True) # 10% is used for thraining data which is approx 2017 data

plt.legend(['Training set (First 90%, approx before 2017)','Test set (Last 10%, approax 2017 and beyond)'])

plt.title('IBM stock price')

plt.show()
# The same LSTM model I would like to test, lets see if the sinus prediction results can be matched

# Note: replace LSTM with GRU or RNN if you want to try those



model = Sequential()

model.add(LSTM(50, return_sequences=True, input_shape=(feature_train.shape[1],1)))

model.add(Dropout(0.2))

model.add(LSTM(100, return_sequences=False))

model.add(Dropout(0.2))

model.add(Dense(1, activation = "linear"))



model.compile(loss='mse', optimizer='adam')



print ('model compiled')
#Train the model

model.fit(feature_train, label_train, batch_size=512, epochs=5, validation_data = (feature_test, label_test))
#Let's use the model and predict the stock

predicted_stock_price = model.predict(feature_test)

plot_results(predicted_stock_price,label_test)
def predict_sequences_multiple(model, data, window_size, prediction_len):

    #Predict sequence of <prediction_len> steps before shifting prediction run forward by <prediction_len> steps

    prediction_seqs = []

    for i in range(int(len(data)/prediction_len)):

        curr_frame = data[i*prediction_len]

        predicted = []

        for j in range(prediction_len):

            predicted.append(model.predict(curr_frame[newaxis,:,:])[0,0])

            curr_frame = curr_frame[1:]

            curr_frame = np.insert(curr_frame, [window_size-1], predicted[-1], axis=0)

        prediction_seqs.append(predicted)

    return prediction_seqs



def plot_results_multiple(predicted_data, true_data, prediction_len):

    fig = plt.figure(facecolor='white')

    ax = fig.add_subplot(111)

    ax.plot(true_data, label='True Data')

    #Pad the list of predictions to shift it in the graph to it's correct start

    for i, data in enumerate(predicted_data):

        padding = [None for p in range(i * prediction_len)]

        plt.plot(padding + data, label='Prediction')

        plt.legend()

    plt.show()



predictions = predict_sequences_multiple(model, feature_test, Enrol_window, 50)

plot_results_multiple(predictions, label_test, 50)  