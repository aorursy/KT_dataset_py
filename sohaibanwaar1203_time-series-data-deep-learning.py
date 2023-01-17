# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
coloum=['ID','symbol','last_price','change','change_percentage','bid_size','bid','offer','offer_size','turnOver','high','low','open','last_volume','total_trades','last_trade','time_stamp']

df=pd.read_csv("../input/Stock_Data_06-11.csv", names=coloum,header=None)





df.drop("ID",axis=1,inplace=True)

df.head()
df.head()
#total companies for which we have to predict stock prices

print(df.symbol.unique())

print("Total Counts of Companies are ",len(df.symbol.unique()))
print("Data type of Time Stamp",type(df.time_stamp[1]))

#we have time steps in string we first have to convert them in DataTime Object

df['time_stamp'] = pd.to_datetime(df['time_stamp'])

print("\nAfter Changing\n")

print("Data type of Time Stamp",type(df.time_stamp[1]))
df["year"]=df["time_stamp"].map(lambda x : x.year)

df["month"]=df["time_stamp"].map(lambda x : x.month)



df["day"]=df["time_stamp"].map(lambda x : x.dayofweek)

df["Hours"]=df["time_stamp"].map(lambda x : x.hour)

df["Minutes"]=df["time_stamp"].map(lambda x : x.minute)

df["Seconds"]=df["time_stamp"].map(lambda x : x.second)

#Seperating Date, year ,month time min second for a deeper dive

df.head(5)
print("We have data of year ",df.year.unique())

print("We have data of month",df.month.unique())

print("We have data of Days",df.day.unique())# from this we can say that we have data of working days only

print("We have data of Hours",df.Hours.unique())

print("We have data of Minutes",df.Minutes.unique())

print("We have data of Seconds",df.Seconds.unique())



# we can conclude that we have data of 2018 of month 6,7,8,9,10,11 which are six and we have data of 

#working days only which are from monday to friday where 0 is reffered as monday and 4 is reffered as friday

# data is gathered in between 9,17 hours
# for now we are just taking one company for simplification 

# e.g we take HBL from these comapnies

df_hbl=df[df.symbol=="HBL"]

print("Total Records",len(df))

print("Length of Records of HBL",len(df_hbl))

# so we can say that we have 54282 records againt every company

df_hbl.head()

# After seperating the data we can see that this dataset have records of every minutes of a day

# in between 9 to 17 hours

# from this we can see that seconds coloum is totally useless for us because from this data we arenot

# able to see seconds to seconds trends. But we can see minutes to minutes trend

df_hbl.index=df_hbl.time_stamp


df_hbl.drop("time_stamp",axis=1,inplace= True)

df_hbl.drop("symbol",axis=1,inplace=True) #all the symbol are same so we dont need it

df_hbl.drop("year",axis=1,inplace=True)# We have only data of 1 year so useless

df_hbl.drop("Seconds",axis=1,inplace=True)# Records are not measured in every second but gathered

# according to the mins thats why Useless

df_hbl.head()

import matplotlib.pyplot as plt

import seaborn as sns

hourly = df_hbl.resample('H').mean()

daily = df_hbl.resample('D').mean()

weekly = df_hbl.resample('W').mean()

monthly = df_hbl.resample('M').mean()

fig, axs = plt.subplots(3,1)



sns.set(rc={'figure.figsize':(16.7,10.27)})

# sns.pointplot(data=hourly,y="Count",x=hourly.index,ax=axs[0]).set_title("Hourly")



sns.pointplot(data=daily,y="last_price",x=daily.index,ax=axs[0]).set_title("Daily")

sns.pointplot(data=weekly,y="last_price",x=weekly.index,ax=axs[1]).set_title("Weekly")

sns.pointplot(data=monthly,y="last_price",x=monthly.index,ax=axs[2]).set_title("Monthly")

plt.show()
# We cannot seperate time series randomly because we have to maintain the sequence to time stamps. Becuase we have to predict

# fureter time counts. 

# split into train and test sets

import matplotlib.pyplot as plt

import pandas

import math

from keras.models import Sequential

from keras.layers import Dense,Dropout

from keras.layers import LSTM

from sklearn.preprocessing import MinMaxScaler

from sklearn.metrics import mean_squared_error



model = Sequential()

dataset=df_hbl.last_price.values

dataset=dataset.reshape(-1,1)

scaler = MinMaxScaler(feature_range=(0, 1))

dataset = scaler.fit_transform(dataset)

train_size = int(len(dataset) * 0.67)

test_size = len(dataset) - train_size

train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]

print(len(train), len(test))
import numpy

def create_dataset(dataset, look_back=1):

	dataX, dataY = [], []

	for i in range(len(dataset)-look_back-1):

		a = dataset[i:(i+look_back), 0]

		dataX.append(a)

		dataY.append(dataset[i + look_back, 0])

	return numpy.asarray(dataX), numpy.asarray(dataY)
from sklearn.preprocessing import MinMaxScaler

look_back=1

trainX, trainY = create_dataset(train, look_back)

testX, testY = create_dataset(test, look_back)

trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))

testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
# define model

model = Sequential()

model.add(LSTM(50, activation='relu', return_sequences=True, input_shape=(1, look_back)))

model.add(LSTM(10, activation='relu'))

model.add(Dense(1))

model.compile(optimizer='adam', loss='mse')

model.fit(trainX, trainY, epochs=3, batch_size=1, verbose=1,validation_data=(testX,testY))
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
trainPredictPlot = numpy.empty_like(dataset)

trainPredictPlot[:, :] = numpy.nan

trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict

# shift test predictions for plotting

testPredictPlot = numpy.empty_like(dataset)

testPredictPlot[:, :] = numpy.nan

testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict

# plot baseline and predictions

plt.plot(scaler.inverse_transform(dataset))

plt.plot(trainPredictPlot)

plt.plot(testPredictPlot)

plt.show()
import numpy as np # linear algebra

import pandas as pd

coloum=['ID','symbol','last_price','change','change_percentage','bid_size','bid','offer','offer_size','turnOver','high','low','open','last_volume','total_trades','last_trade','time_stamp']

df=pd.read_csv("../input/Stock_Data_06-11.csv", names=coloum,header=None)





df.drop("ID",axis=1,inplace=True)



df['time_stamp'] = pd.to_datetime(df['time_stamp'])

df.dtypes
companies=df.symbol.unique()

dataframe_collection = {} 



for i in companies:

    df_temp=df[df.symbol==i]

    close_price=df_temp.last_price.values

    time_stamp=df_temp.time_stamp.values

    

    dataframe_collection[i] = pd.DataFrame(list(zip(close_price,time_stamp)), columns=["last_price", "timestamp"])

    dataframe_collection[i].index=dataframe_collection[i].timestamp

    dataframe_collection[i].drop("timestamp",axis=1,inplace=True)

    print("Values of {}".format(i),len(dataframe_collection[i]))

#here we make list of datafrma to have seperate dataframe for every company because every company has its own trends



    
def LSTMmodel(look_back=1):

    # define model

    model = Sequential()

    model.add(LSTM(25, activation='relu', return_sequences=True, input_shape=(1, look_back)))

    model.add(LSTM(10, activation='relu'))

    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mse')

    return model



import numpy

def create_dataset(dataset, look_back=1):

	dataX, dataY = [], []

	for i in range(len(dataset)-look_back-1):

		a = dataset[i:(i+look_back), 0]

		dataX.append(a)

		dataY.append(dataset[i + look_back, 0])

	return numpy.asarray(dataX), numpy.asarray(dataY)

    
def save_model(model,name):

    model_json = model.to_json()

    with open("{}.json".format(name), "w") as json_file:

        json_file.write(model_json)

    # serialize weights to HDF5

    model.save_weights("{}.h5".format(name))

    print("Saved model of {} to disk".format(i))


def Learning(df,look_back,name):

    

    dataset=df.last_price.values

    dataset=dataset.reshape(-1,1)

    scaler = MinMaxScaler(feature_range=(0, 1))

    dataset = scaler.fit_transform(dataset)

    train_size = int(len(dataset) * 0.67)

    test_size = len(dataset) - train_size

    train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]

    print("Length of Train: {} and Test: {}".format(len(train), len(test)))

    trainX, trainY = create_dataset(train, look_back)

    testX, testY = create_dataset(test, look_back)

    trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))

    testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

    model=LSTMmodel(look_back)

    model.fit(trainX, trainY, epochs=1, batch_size=1, verbose=1,validation_data=(testX,testY))

    

    # saving model to disk

    save_model(model,name)

    

    #Prediction Graphs on Validation

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

    trainPredictPlot = numpy.empty_like(dataset)

    trainPredictPlot[:, :] = numpy.nan

    trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict

    # shift test predictions for plotting

    testPredictPlot = numpy.empty_like(dataset)

    testPredictPlot[:, :] = numpy.nan

    testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict

    # plot baseline and predictions

    plt.plot(scaler.inverse_transform(dataset))

    plt.plot(trainPredictPlot)

    plt.plot(testPredictPlot)

    plt.show()
from sklearn.preprocessing import MinMaxScaler

import math

from keras.models import Sequential

from keras.layers import Dense,Dropout

from keras.layers import LSTM

from sklearn.preprocessing import MinMaxScaler

from sklearn.metrics import mean_squared_error

import matplotlib.pyplot as plt

look_back=1

for i in companies:

    print("Prediction For ",i)

    Learning(dataframe_collection[i],look_back,i)