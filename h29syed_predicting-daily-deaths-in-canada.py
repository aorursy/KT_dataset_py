# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from keras.models import Sequential

from keras.layers import Dense

from keras.layers import LSTM

from sklearn.preprocessing import MinMaxScaler

from sklearn.metrics import mean_squared_error



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#Read data



df_death = pd.read_csv("/kaggle/input/ece657aw20asg4coronavirus/time_series_covid19_deaths_global.csv")

df_confirmed = pd.read_csv("/kaggle/input/ece657aw20asg4coronavirus/time_series_covid19_confirmed_global.csv")

df_recovered = pd.read_csv("/kaggle/input/ece657aw20asg4coronavirus/time_series_covid19_recovered_global.csv")



canada_deaths = df_death.iloc[35:46]

canada_confirmed = df_confirmed.iloc[35:46]

canada_recovered = df_recovered.iloc[35:46]

total_canada_deaths = []

total_canada_confirmed = []

total_canada_recovered = []

days_since_jan22 = []

count = 0

for i in range(4,91):

    total_canada_deaths.append(np.array(canada_deaths[canada_deaths.columns[i]].sum()))

    total_canada_confirmed.append(np.array(canada_confirmed[canada_confirmed.columns[i]].sum()))

    total_canada_recovered.append(np.array(canada_recovered[canada_recovered.columns[i]].sum()))

    days_since_jan22.append(count)

    count = count + 1



data = []



for i in range(87):

    data.append([days_since_jan22[i], total_canada_confirmed[i], total_canada_recovered[i], total_canada_deaths[i]])

    



#total_canada_deaths = pd.DataFrame(total_canada_deaths) 



#print(total_canada_deaths)



data = pd.DataFrame(data, columns = ['Days_since_Jan_22','Cases', 'Recovered','Deaths']) 

print(data)

#our_df= pd.DataFrame(data, columns = ['Number of days since Jan 22', 'Deaths'])
#Plot the number od deaths in Canada since Jan 22



plt.plot(data['Days_since_Jan_22'], data['Deaths'],label = "Actual")

plt.legend()

plt.title('Total Number of COVID-19 Deaths in Canada')

plt.xlabel('Day')

plt.ylabel('Number of Deaths')

plt.grid()
data = data.values

#data = data.astype('float32')



# normalize the dataset

#scaler = MinMaxScaler(feature_range=(0, 1))

#data = scaler.fit_transform(data)



#Split into train and test sets

train_size = int(len(data) * 0.67)

test_size = len(data) - train_size

train, test = data[0:train_size,:], data[train_size:len(data),:]

print(len(train), len(test))



#Predict using linear regression





#y = y.reshape(1,-1)



#lm = linear_model.LinearRegression()

#model = lm.fit(X,y)



#predictions = lm.predict(X)

#print(predictions)

#plt.plot(X, predictions, label = "Predicted Deaths")

#plt.grid()

def create_dataset(dataset, look_back=1):

    dataX, dataY = [], []

    for i in range(len(dataset)-look_back-1):

        a = dataset[i:(i+look_back), 0]

        dataX.append(a)

        dataY.append(dataset[i + look_back, 0])

    return np.array(dataX), np.array(dataY)
look_back = 10

trainX, trainY = create_dataset(train, look_back)

testX, testY = create_dataset(test, look_back)


# reshape input to be [samples, time steps, features]

trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))

testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
# create and fit the LSTM network

model = Sequential()

model.add(LSTM(4, input_shape=(1, look_back)))

model.add(Dense(1))

model.compile(loss='mean_squared_error', optimizer='adam')

model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=2)
# make predictions

trainPredict = model.predict(trainX)

testPredict = model.predict(testX)



print(trainPredict)

print(testPredict)