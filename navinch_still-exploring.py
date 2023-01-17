# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
ar = pd.read_csv('../input/air_reserve.csv')

asi = pd.read_csv('../input/air_store_info.csv')

avd = pd.read_csv('../input/air_visit_data.csv')

di = pd.read_csv('../input/date_info.csv')

hr = pd.read_csv('../input/hpg_reserve.csv')

hsi = pd.read_csv('../input/hpg_store_info.csv')

sid = pd.read_csv('../input/store_id_relation.csv')
#air_8093d0b565e9dbdf hpg_874415e6e7ccfe13
store = avd[avd['air_store_id'] == 'air_8093d0b565e9dbdf']

store = store.merge(di, left_on='visit_date', right_on='calendar_date')

store.drop(['calendar_date', 'day_of_week', 'air_store_id'], axis=1, inplace=True)

store.set_index('visit_date', inplace=True)
store.head()
def rmsle(y, y_pred):

    assert len(y) == len(y_pred)

    to_sum = [(math.log(pred + 1) - math.log(y[i] + 1)) ** 2.0 for i,pred in enumerate(y_pred)]

    return (sum(to_sum) * (1.0/len(y))) ** 0.5
import math

from keras.models import Sequential

from keras.layers import Dense

from keras.layers import LSTM

from sklearn.preprocessing import MinMaxScaler

from sklearn.metrics import mean_squared_error

# convert an array of values into a dataset matrix

def create_dataset(dataset, look_back=1):

    dataX, dataY = [], []

    for i in range(len(dataset)-look_back-1):

        a = dataset[i:(i+look_back), 0]

        dataX.append(a)

        dataY.append(dataset[i + look_back, 0])

    return np.array(dataX), np.array(dataY)

# fix random seed for reproducibility

np.random.seed(7)

# load the dataset

dataset = store['visitors'].values

dataset = dataset.astype('int32')

# normalize the dataset

scaler = MinMaxScaler(feature_range=(0, 1))

dataset = scaler.fit_transform(dataset.reshape(-1, 1))

# split into train and test sets

train_size = int(len(dataset) * 0.8)

test_size = len(dataset) - train_size

train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]

# reshape into X=t and Y=t+1

look_back = 7

trainX, trainY = create_dataset(train, look_back)

testX, testY = create_dataset(test, look_back)

# reshape input to be [samples, time steps, features]

trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))

testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

# create and fit the LSTM network

model = Sequential()

model.add(LSTM(100, input_shape=(1, look_back)))

model.add(Dense(7))

model.add(Dense(1))

model.compile(loss='mean_squared_error', optimizer='adam')

model.fit(trainX, trainY, epochs=15, batch_size=1, verbose=2)

# make predictions

trainPredict = model.predict(trainX)

testPredict = model.predict(testX)

# invert predictions

trainPredict = scaler.inverse_transform(trainPredict)

trainY = scaler.inverse_transform([trainY])

testPredict = scaler.inverse_transform(testPredict)

testY = scaler.inverse_transform([testY])

# calculate root mean squared error

trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))

print('Train Score: %.2f RMSE' % (trainScore))

testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))

print('Test Score: %.2f RMSE' % (testScore))

# shift train predictions for plotting

trainPredictPlot = np.empty_like(dataset)

trainPredictPlot[:, :] = np.nan

trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict

# shift test predictions for plotting

testPredictPlot = np.empty_like(dataset)

testPredictPlot[:, :] = np.nan

testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict

# plot baseline and predictions

plt.plot(scaler.inverse_transform(dataset))

plt.plot(trainPredictPlot)

plt.plot(testPredictPlot)

plt.show()
ratios = [x/y for x,y in zip(testPredict, testY)]
plt.plot(ratios[0])

plt.show()
ones = [x for x in ratios[0] if (x>0.6) & (x<1.3)]

len(ones)/len(ratios[0])
rmsle(testY.T, testPredict)