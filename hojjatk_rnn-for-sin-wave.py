import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import os



print(os.listdir("../input"))
# Load the data

dataset = pd.read_csv('../input/Sin Wave Data Generator.csv')

dataset.head(5)
dataset["Wave"][:].plot(figsize=(16,4),legend=False)
def normalise_windows(window_data):

    # A support function to normalize a dataset

    normalised_data = []

    for window in window_data:

        normalised_window = [((float(p) / float(window[0])) - 1) for p in window]

        normalised_data.append(normalised_window)

    return normalised_data



def load_data(dataset, column, seq_len, normalise_window):

    # A support function to help prepare datasets for an RNN/LSTM/GRU

    data = dataset.loc[:,column]

    sequence_length = seq_len + 1

    result = []

    

    for index in range(len(data) - sequence_length):

        result.append(data[index: index + sequence_length])

    

    if normalise_window:

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
Enrol_window = 100

feature_train, label_train, feature_test, label_test = load_data(dataset, 'Wave', Enrol_window, False)

print ('enrol window set to', Enrol_window )

print ('Datasets generated')



# If not shuffle, then the following will be the case

# label_train[0]) = feature_train[100][0]
from keras.layers.core import Dense, Activation, Dropout

from keras.layers.recurrent import LSTM, GRU

from keras.models import Sequential

from keras import optimizers



# The LSTM model I would like to test

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

model.fit(feature_train, label_train, batch_size=512, epochs=10, validation_data = (feature_test, label_test))
from numpy import newaxis



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



#Let's use the model and predict the wave

predictions= predict_sequence_full(model, feature_test, Enrol_window)

plot_results(predictions,label_test)