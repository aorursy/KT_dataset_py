import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import gc;gc.collect

import datetime



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

#holidays_events = pd.read_csv('../input/holidays_events.csv')

#items = pd.read_csv('../input/items.csv')

#oil = pd.read_csv('../input/oil.csv')

sample_submission = pd.read_csv('../input/sample_submission.csv')

#stores = pd.read_csv('../input/stores.csv')

test = pd.read_csv('../input/test.csv')

train = pd.read_csv('../input/train.csv')

#transactions = pd.read_csv('../input/transactions.csv')

train["date"] =  pd.to_datetime(train["date"])

train_2016 = train[train["date"].dt.year == 2016]

date_list=np.unique(train_2016["date"])

del train; gc.collect()

#train_2016["Year"]=train_2016["date"].dt.year

#train_2016["Month"]=train_2016["date"].dt.month

#train_2016["Day"]=train_2016["date"].dt.day

#date_list


#train_2016.set_index("date",inplace=True)

train_2016.drop("id",axis=1,inplace=True)

train_2016.drop("onpromotion",axis=1,inplace=True)

train_2016.fillna(0, inplace=True)

print(train_2016.head(5))



#train_2016.to_csv('train_2016.csv')
memo = train_2016.memory_usage(index=True).sum()

print(memo/ 1024**2," MB")

print(train_2016.dtypes)
train_2016['store_nbr'] = train_2016['store_nbr'].astype(np.int8)

train_2016['item_nbr'] = train_2016['item_nbr'].astype(np.int32)

train_2016['unit_sales'] = train_2016['unit_sales'].astype(np.int8)

memo = train_2016.memory_usage(index=True).sum()

print(memo/ 1024**2," MB")

train_2016.head()
train_2016_25= train_2016[train_2016["store_nbr"] == 25]

train_2016_25_105574= train_2016_25[train_2016_25["item_nbr"] == 105574]

train_2016_25_105574.drop("store_nbr",axis=1,inplace=True)

train_2016_25_105574.drop("item_nbr",axis=1,inplace=True)

train_2016_25_105574.plot(figsize=(15, 6))

plt.show()
df_date=pd.DataFrame(date_list)

df_date = df_date.rename(columns={0: 'date'})

df_date.head()
df = pd.merge(df_date, train_2016_25_105574, how='left', on=['date'])

df.fillna(0,inplace=True)



df.head()
df.plot(figsize=(15, 6))

plt.show()

df.describe()
import numpy

import matplotlib.pyplot as plt

import pandas

import math

from keras.models import Sequential

from keras.layers import Dense,Dropout

from keras.layers import LSTM

from sklearn.preprocessing import MinMaxScaler

from sklearn.metrics import mean_squared_error
# fix random seed for reproducibility

numpy.random.seed(7)





# split into train and test sets

train, test =df[df["date"].dt.month <= 11], df[df["date"].dt.month > 11]

train=train.set_index("date")

test=test.set_index("date")

df=df.set_index("date")

train=train.values

test=test.values

df=df.values

def create_dataset(dataset, look_back=1):

    dataX, dataY = [], []

    for i in range(len(dataset)-look_back-1):

        a = dataset[i:(i+look_back), 0]

        dataX.append(a)

        dataY.append(dataset[i + look_back, 0])

    return numpy.array(dataX), numpy.array(dataY)
# reshape into X=t and Y=t+1

look_back = 4

trainX, trainY = create_dataset(train, look_back)

testX, testY = create_dataset(test, look_back)


# reshape input to be [samples, time steps, features]

trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))

testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
# create and fit the LSTM network

model = Sequential()

model.add(LSTM(4, input_shape=(1, look_back)))

model.add(Dense(10))

model.add(Dropout(0.2))

model.add(Dense(1))

model.compile(loss='mean_squared_error', optimizer='adam')

model.fit(trainX, trainY, validation_data=(testX, testY),epochs=25, batch_size=1, verbose=2)
# make predictions

trainPredict = model.predict(trainX)

testPredict = model.predict(testX)
# shift train predictions for plotting

trainPredictPlot = numpy.empty_like(df)

trainPredictPlot[:, :] = numpy.nan

trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict

# shift test predictions for plotting

testPredictPlot = numpy.empty_like(df)

testPredictPlot[:, :] = numpy.nan

testPredictPlot[len(trainPredict)+(look_back*2)+1:len(df)-1, :] = testPredict

# plot baseline and predictions



plt.plot(df)

plt.plot(trainPredictPlot)

plt.plot(testPredictPlot)

plt.show()
result=pd.DataFrame(testPredict)

result["testY"]=testY

result