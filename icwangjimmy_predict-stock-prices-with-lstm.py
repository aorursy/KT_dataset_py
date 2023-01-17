import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from subprocess import check_output

from keras.layers.core import Dense, Activation, Dropout

from keras.layers.recurrent import LSTM

from keras.models import Sequential

from sklearn.cross_validation import  train_test_split

import time #helper libraries

from sklearn.preprocessing import MinMaxScaler

import matplotlib.pyplot as plt

from numpy import newaxis
prices_dataset =  pd.read_csv('../input/prices.csv', header=0)

prices_dataset.head()
wltw = prices_dataset[prices_dataset['symbol']=='WLTW']

wltw.shape
wltw_stock_prices = wltw.close.values.astype('float32')



plt.plot(wltw_stock_prices)

plt.show()



wltw_stock_prices=wltw_stock_prices.reshape(-1, 1)

scaler = MinMaxScaler(feature_range=(0, 1))

wltw_stock_prices = scaler.fit_transform(wltw_stock_prices)
train_size = int(len(wltw_stock_prices) * 0.67)

test_size = len(wltw_stock_prices) - train_size

train, test = wltw_stock_prices[0:train_size,:], wltw_stock_prices[train_size:len(wltw_stock_prices),:]

print(len(train), len(test))
# convert an array of values into a dataset matrix

def create_dataset(dataset, look_back=1):

	dataX, dataY = [], []

	for i in range(len(dataset)-look_back-1):

		a = dataset[i:(i+look_back), 0]

		dataX.append(a)

		dataY.append(dataset[i + look_back, 0])

	return np.array(dataX), np.array(dataY)
# reshape into X=t and Y=t+1

look_back = 5

trainX, trainY = create_dataset(train, look_back)

testX, testY = create_dataset(test, look_back)
trainX.shape
trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))

testX = np.reshape(testX, (testX.shape[0], testX.shape[1], 1))
trainX.shape
testX.shape
#Step 2 Build Model

model = Sequential()



model.add(LSTM(

    input_dim=1,

    input_length=5,

    output_dim=50,

    return_sequences=True))

model.add(Dropout(0.2))



model.add(LSTM(

    100,

    return_sequences=False))

model.add(Dropout(0.2))



model.add(Dense(

    output_dim=1))

model.add(Activation('linear'))



start = time.time()

model.compile(loss='mse', optimizer='rmsprop')

print ('compilation time : ', time.time() - start)
model.summary()
model.get_config()
model.fit(

    trainX,

    trainY,

    batch_size=128,

    nb_epoch=10,

    validation_split=0.10)
def plot_results_multiple(predicted_data, true_data, prediction_len):

    fig = plt.figure(facecolor='white', figsize=(16,8))

    ax = fig.add_subplot(111)

    ax.plot(true_data, label='True Data')

    #print ('yo')

    #Pad the list of predictions to shift it in the graph to it's correct start

    for i, data in enumerate(predicted_data):

        padding = [None for p in range(i * prediction_len)]

        #

        plt.plot(padding + data, label='Prediction')

        plt.plot(padding + data, 'b^')

    plt.legend()

    plt.show()

    

def predict_sequences_multiple(model, data, window_size, prediction_len):

    #Predict sequence of 5 steps before shifting prediction run forward by 5 steps

    #Predict length is how many points I want to predict

    #window_size is how many time steps I need for the prediction

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



predictions = predict_sequences_multiple(model, testX, 5, 1)
len(predictions)
plot_results_multiple(predictions, testY, 1)