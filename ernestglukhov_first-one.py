from __future__ import print_function
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from sklearn.model_selection import  train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import math
from sklearn.preprocessing import StandardScaler

# to not display the warnings of tensorflow
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
# parameters to be set ("optimum" hyperparameters obtained from grid search):
look_back = 7
epochs = 10
batch_size = 32
# fix random seed for reproducibility
np.random.seed(7)
# read all prices using panda
prices_dataset =  pd.read_csv('../input/train.csv', header=0)
test_dataset =  pd.read_csv('../input/test.csv', header=0)
def create_dataset(dataset, look_back):
    inds = []
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        inds.append(i)
        a = dataset[i:(i+look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
#     print(inds)
    return np.array(dataX), np.array(dataY)
prices_dataset.head()
prices_dataset.asset.unique()
res_all = np.array([]).reshape(-1, 1)
for asset in prices_dataset.asset.unique():
    apple = prices_dataset[prices_dataset['asset']==asset]

    apple_stock_prices = apple.close.values.astype('float32')

    apple_stock_prices = apple_stock_prices.reshape(len(apple_stock_prices), 1)

    # normalize the dataset
    scaler = MinMaxScaler(feature_range=(0, 1))
    apple_stock_prices = scaler.fit_transform(apple_stock_prices)

    # split data into training set and test set
    train_size = int(len(apple_stock_prices) * 0.67)
    test_size = len(apple_stock_prices) - train_size
    train, test = apple_stock_prices[0:train_size,:], apple_stock_prices[train_size:len(apple_stock_prices),:]

    print('Split data into training set and test set... Number of training samples/ test samples:', len(train), len(test))

    # convert an array of values into a time series dataset 
    # in form 
    #                     X                     Y
    # t-look_back+1, t-look_back+2, ..., t     t+1



    # convert Apple's stock price data into time series dataset
    trainX, trainY = create_dataset(train, look_back)
    testX, testY = create_dataset(test, look_back)

    # reshape input of the LSTM to be format [samples, time steps, features]
    trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
    testX = np.reshape(testX, (testX.shape[0], testX.shape[1], 1))

    # create and fit the LSTM network
    model = Sequential()
    model.add(LSTM(4, input_shape=(look_back, 1)))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    model.fit(trainX, trainY, nb_epoch=epochs, batch_size=batch_size, verbose=0)

    # make predictions
    trainPredict = model.predict(trainX)
    testPredict = model.predict(testX)

    # invert predictions and targets to unscaled
    trainPredict = scaler.inverse_transform(trainPredict)
    trainY = scaler.inverse_transform([trainY])
    testPredict = scaler.inverse_transform(testPredict)
    testY = scaler.inverse_transform([testY])

    # calculate root mean squared error
    trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
    print('Train Score: %.2f RMSE' % (trainScore))
    testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
    print('Test Score: %.2f RMSE' % (testScore))

    # shift predictions of training data for plotting
    trainPredictPlot = np.empty_like(apple_stock_prices)
    trainPredictPlot[:, :] = np.nan
    trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict

    # shift predictions of test data for plotting
    testPredictPlot = np.empty_like(apple_stock_prices)
    testPredictPlot[:, :] = np.nan
    testPredictPlot[len(trainPredict)+(look_back*2)+1:len(apple_stock_prices)-1, :] = testPredict

    # plot baseline and predictions
    plt.plot(scaler.inverse_transform(apple_stock_prices))
    plt.plot(trainPredictPlot)
    plt.plot(testPredictPlot)
    plt.show()

    ## FOR predictions:



    apple = test_dataset[test_dataset['asset']==asset]

    apple_stock_prices = apple.close.values.astype('float32')

    apple_stock_prices = apple_stock_prices.reshape(len(apple_stock_prices), 1)

    # normalize the dataset
    scaler = MinMaxScaler(feature_range=(0, 1))
    apple_stock_prices = scaler.fit_transform(apple_stock_prices)

    testX, testY = create_dataset(apple_stock_prices, look_back)

    testX = np.reshape(testX, (testX.shape[0], testX.shape[1], 1))

    testPredict = model.predict(testX)

    
    
    price = testPredict[1:]/testPredict[:-1] - 1
    price = np.append([0], price).reshape(-1, 1)
    z = 1440 - price.shape[0]
    res = np.append(price, np.zeros(z)).reshape(-1, 1)
#     res = np.append([0], np.ediff1d(price)).reshape(-1, 1)
    
    res_all = np.vstack((res_all,res))
scaler = MinMaxScaler(feature_range=(-1, 1))
res_all = scaler.fit_transform(res_all)
res_all[res_all<-1]=-1
res_all[res_all>1]=1
out_df = pd.DataFrame(res_all, columns=['expected'])
out_df.to_csv('Elastic_net.csv',index = True)
df_res = pd.read_csv('Elastic_net.csv')
df_res.columns = ['id', 'expected']
df_res.to_csv('submit.csv', index=False)