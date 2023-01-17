import numpy as np

import pandas as pd

from matplotlib import pyplot as plt

from sklearn.preprocessing import MinMaxScaler
dt_parser = lambda x, y: pd.datetime.strptime(str(x) + str(y), '%d/%m/%Y%H:%M:%S')

df = pd.read_csv('../input/householdpowerconsumption/household_power_consumption.txt', sep = ';',

                parse_dates={'dt' : ['Date', 'Time']}, date_parser = dt_parser, index_col='dt',

                low_memory=False)

df.head()
df.info()
for col in df:

    print(df[col].value_counts(dropna=False))
df.replace('?', np.nan, inplace=True)

for col in df:

    df[col] = df[col].astype(float)

df.fillna(df.mean(), inplace=True)
# Check

df.isnull().sum()
df.describe()
def plot_mean_std(data):

    plt.style.use('seaborn')

    fig, (ax1, ax2) = plt.subplots(2,1, figsize=(10,10))

    ax1.plot(data.index.to_pydatetime(), data['mean'], color= 'b')

    ax2.plot(data.index.to_pydatetime(), data['std'], color= 'g')

    
r = df['Global_active_power'].resample('D').agg(['mean', 'std'])

plot_mean_std(r)
r = df['Global_reactive_power'].resample('D').agg(['mean', 'std'])

plot_mean_std(r)
r = df['Global_intensity'].resample('D').agg(['mean', 'std'])

plot_mean_std(r)
# resample by day too messy. 

sub1 = df['Sub_metering_1'].resample('2W').agg(['mean', 'std'])

plot_mean_std(sub1)
sub2 = df['Sub_metering_2'].resample('2W').agg(['mean', 'std'])

plot_mean_std(sub2)
sub3 = df['Sub_metering_3'].resample('2W').agg(['mean', 'std'])

plot_mean_std(sub3)
volt = df['Voltage'].resample('d').agg(['mean', 'std'])

plot_mean_std(volt)
# Resample because of the high computational load.

df = df.resample('h').mean()

print(df.shape)
values = df.values

scaler = MinMaxScaler(feature_range=(0, 1))

df[:] = scaler.fit_transform(values)

df.head()
# The number of previous timesteps included.

prev_steps = 10

# The gap size of previous timesteps. Higher makes the model sees longer while eliminating more rows.

step_gap = 1



for i in range(prev_steps):

    # t - (i+1), i+1 = 1, ..., prev_steps

    df0 = df.iloc[:,:7].shift(step_gap)

    if (i!=0):

        df0.rename(columns = {x:x[:-3] +'-%02d' % (i+1) for x in df0.columns}, inplace=True)

    else:

        df0.rename(columns = {x:x +'-%02d'% (i+1) for x in df0.columns}, inplace=True)

        

    df = pd.concat([df0,df], axis = 1)



# t+1

df['GAP next'] = df['Global_active_power'].shift(-1)    

df.dropna(inplace = True)

df.head()
train = df[:'2009-07-01 00:00:00'].values 

test = df['2009-07-01 00:00:00':].values

print(train.shape, test.shape)
train_X, train_y = train[:,:-1], train[:,-1]

test_X, test_y = test[:,:-1], test[:,-1]
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))

test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
import keras

from keras.layers import Dense

from keras.models import Sequential

from keras.optimizers import SGD

from keras.layers import LSTM

from keras.layers import Dropout

from sklearn.metrics import mean_squared_error
model = Sequential()

model.add(LSTM(64, input_shape=(train_X.shape[1], train_X.shape[2]),return_sequences=True))

model.add(Dropout(0.4))

model.add(LSTM(64))

model.add(Dropout(0.4))

model.add(Dense(1))

model.compile(loss='mean_squared_error', optimizer='adam')

model.summary()
model.fit(train_X, train_y, epochs=200, validation_data=(test_X, test_y), shuffle=False)

y_hat = model.predict(test_X)
test_X.shape
# Get back only the last 7 columns for inversion. 

test_X = test_X[:,:,-7:].reshape((test_X.shape[0], 7))



# invert scaling for y_hat

y_hat_x = np.concatenate((y_hat, test_X[:, 1:]), axis=1)

y_hat_x = scaler.inverse_transform(y_hat_x)

inv_y_hat = y_hat_x[:,0]



# invert scaling for the original test value

test_y = test_y.reshape((test_y.shape[0], 1))

test_y_x = np.concatenate((test_y, test_X[:, 1:]), axis=1)

test_y_x = scaler.inverse_transform(test_y_x)

inv_y = test_y_x[:,0]

# calculate RMSE

rmse = np.sqrt(mean_squared_error(inv_y, inv_y_hat))

print('Test RMSE: %.3f' % rmse)
x = range(inv_y.shape[0])

plt.plot(x, inv_y, label="actual")

plt.plot(x, inv_y_hat, label="prediction")

plt.ylabel('Global_active_power', size = 15)

plt.xlabel('Time step', size = 15)

plt.legend(fontsize=15)

plt.show()