#Import Dependencies



import numpy as np

from numpy import nan

import pandas as pd

import matplotlib.pyplot as plt



from sklearn.metrics import mean_squared_error

from sklearn.preprocessing import MinMaxScaler



from tensorflow.keras import Sequential

from tensorflow.keras.layers import LSTM, Dense
data = pd.read_csv('../input/power-consumption-of-house/power_consumption_of_house.txt', sep = ';', parse_dates = True, low_memory = False)



data['date_time'] = data['Date'].str.cat(data['Time'], sep= ' ')

data.drop(['Date', 'Time'], inplace= True, axis = 1)



data.set_index(['date_time'], inplace=True)

data.replace('?', nan, inplace=True)

data = data.astype('float')

data.head()
#First check how many values are null

np.isnan(data).sum()



#fill the null value



def fill_missing(data):

    one_day = 24*60

    for row in range(data.shape[0]):

        for col in range(data.shape[1]):

            if np.isnan(data[row, col]):

                data[row, col] = data[row-one_day, col]



fill_missing(data.values)



#Again check the data after filling the value

np.isnan(data).sum()
data.describe()

data.shape
data.head()
# Converting the index as date

data.index = pd.to_datetime(data.index)
data = data.resample('D').sum()
data.head()


fig, ax = plt.subplots(figsize=(18,18))



for i in range(len(data.columns)):

    plt.subplot(len(data.columns), 1, i+1)

    name = data.columns[i]

    plt.plot(data[name])

    plt.title(name, y=0, loc = 'right')

    plt.yticks([])

plt.show()

fig.tight_layout()
years = ['2007', '2008', '2009', '2010']



fig, ax = plt.subplots(figsize=(18,18))

for i in range(len(years)):

    plt.subplot(len(years), 1, i+1)

    year = years[i]

    active_power_data = data[str(year)]

    plt.plot(active_power_data['Global_active_power'])

    plt.title(str(year), y = 0, loc = 'left')

plt.show()

fig.tight_layout()
fig, ax = plt.subplots(figsize=(18,18))



for i in range(len(years)):

    plt.subplot(len(years), 1, i+1)

    year = years[i]

    active_power_data = data[str(year)]

    active_power_data['Global_active_power'].hist(bins = 200)

    plt.title(str(year), y = 0, loc = 'left')

plt.show()

fig.tight_layout()
# for full data



fig, ax = plt.subplots(figsize=(18,18))



for i in range(len(data.columns)):

    plt.subplot(len(data.columns), 1, i+1)

    name = data.columns[i]

    data[name].hist(bins=200)

    plt.title(name, y=0, loc = 'right')

    plt.yticks([])

plt.show()

fig.tight_layout()
data_train = data.loc[:'2009-12-31', :]['Global_active_power']

data_train.head()
data_test = data['2010']['Global_active_power']

data_test.head()
data_train.shape
data_test.shape
data_train = np.array(data_train)

print(data_train)



X_train, y_train = [], []

for i in range(7, len(data_train)-7):

    X_train.append(data_train[i-7:i])

    y_train.append(data_train[i:i+7])

    

X_train, y_train = np.array(X_train), np.array(y_train)

X_train.shape, y_train.shape
pd.DataFrame(X_train).head()
x_scaler = MinMaxScaler()

X_train = x_scaler.fit_transform(X_train)



y_scaler = MinMaxScaler()

y_train = y_scaler.fit_transform(y_train)
X_train = X_train.reshape(1098, 7, 1)
X_train.shape
model = Sequential()

model.add(LSTM(units = 200, activation = 'relu', input_shape=(7,1)))

model.add(Dense(7))



model.compile(loss='mse', optimizer='adam')
model.summary()
model.fit(X_train, y_train, epochs = 100)
data_test = np.array(data_test)
X_test, y_test = [], []



for i in range(7, len(data_test)-7):

    X_test.append(data_test[i-7:i])

    y_test.append(data_test[i:i+7])
X_test, y_test = np.array(X_test), np.array(y_test)


X_test = x_scaler.transform(X_test)

y_test = y_scaler.transform(y_test)
X_test = X_test.reshape(331,7,1)
X_test.shape
y_pred = model.predict(X_test)
y_pred = y_scaler.inverse_transform(y_pred)

y_pred
y_true = y_scaler.inverse_transform(y_test)

y_true


def evaluate_model(y_true, y_predicted):

    scores = []

    

    #calculate scores for each day

    for i in range(y_true.shape[1]):

        mse = mean_squared_error(y_true[:, i], y_predicted[:, i])

        rmse = np.sqrt(mse)

        scores.append(rmse)

    

    #calculate score for whole prediction

    total_score = 0

    for row in range(y_true.shape[0]):

        for col in range(y_predicted.shape[1]):

            total_score = total_score + (y_true[row, col] - y_predicted[row, col])**2

    total_score = np.sqrt(total_score/(y_true.shape[0]*y_predicted.shape[1]))

    

    return total_score, scores
evaluate_model(y_true, y_pred)