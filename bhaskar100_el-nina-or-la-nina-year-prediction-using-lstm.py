# Import all required libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline



import keras as kr

import sklearn

import math

from keras.models import Sequential

from keras.layers import Dense, Activation, LSTM

from sklearn.preprocessing import MinMaxScaler

from sklearn.metrics import mean_squared_error



import itertools

import warnings

warnings.filterwarnings('ignore')
data = pd.read_csv('../input/El-Nino.csv', sep = '\t')
data.head()
cols = ['Year','Janauary','February','March','April','May','June','July','August','September','October','November','December']
data.columns = cols
data.head(10)
data.set_index('Year', inplace = True)

data.head()
data1 = data.transpose()

data1
dates = pd.date_range(start = '1950-01', freq = 'MS', periods = len(data1.columns)*12)

dates
data_np = data1.transpose().as_matrix()

shape = data_np.shape

data_np
data_np = data_np.reshape((shape[0] * shape[1], 1))

data_np.shape
df = pd.DataFrame({'Mean' : data_np[:,0]})

df.set_index(dates, inplace = True)

df.head()
plt.figure(figsize = (15,5))

plt.plot(df.index, df['Mean'])

plt.title('Yearly vs Monthly Mean')

plt.xlabel('Year')

plt.ylabel('Mean across Month')
dataset = df.values

dataset.shape
train = dataset[0:696,:]

test = dataset[696:,:]
print("Original data shape:",dataset.shape)

print("Train shape:",train.shape)

print("Test shape:",test.shape)
# Converting the data into MinMax Scaler because to avoid any outliers present in our dataset

scaler = MinMaxScaler(feature_range = (0,1))

scaled_data = scaler.fit_transform(dataset)

scaled_data.shape
x_train, y_train = [], []

for i in range(60,len(train)):

    x_train.append(scaled_data[i-60:i,0])

    y_train.append(scaled_data[i,0])

x_train, y_train = np.array(x_train), np.array(y_train)
#x_train shape

x_train.shape
#y_train shape

y_train.shape
x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))

x_train.shape
 # Creating and fitting the model

model = Sequential()

model.add(LSTM(units = 50, return_sequences = True, input_shape = (x_train.shape[1],1)))

model.add(LSTM(units = 50))

model.add(Dense(1))
model.compile(loss = 'mean_squared_error', optimizer = 'adam')

model.fit(x_train, y_train, epochs=10, batch_size = 1, verbose = 2)
# Now Let's perform same operations that are done on train set

inputs = df[len(df) - len(test) - 60:].values

inputs = inputs.reshape(-1,1)

inputs = scaler.transform(inputs)
X_test = []

for i in range(60,inputs.shape[0]):

    X_test.append(inputs[i-60:i,0])

X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))

Mean = model.predict(X_test)

Mean1 = scaler.inverse_transform(Mean)
# Check for the RMS error between test set and Mean1 predicted values

rms=np.sqrt(np.mean(np.power((test-Mean1),2)))

rms
#plotting the train, test and forecast data

train = df[:696]

test = df[696:]

test['Predictions'] = Mean1



plt.figure(figsize=(15,5))

plt.plot(train['Mean'])

plt.plot(test['Mean'], color = 'black')

plt.plot(test['Predictions'], color = 'orange')

plt.xlabel('Years')

plt.ylabel('Mean')

plt.title('Forecasting on Actual data')

# Here we are taking steps as 2, means we have taken test size as 120 that is step=1.  

#steps=2 means taking 120 test values and 120 future values i.e next 10 year values from test data

trainpred = model.predict(X_test,steps=2)
trainpred.shape
pred = scaler.inverse_transform(trainpred)
 # Total predicted values are 240, but now I'm printing only first 24 values

pred[0:24] 
test.head()
# Now printing the test Accuracy

testScore = math.sqrt(mean_squared_error(test['Mean'], trainpred[:120,0]))*100

print('Accuracy Score: %.2f' % (testScore))
# Now consider which year we want to predict the value

# Here enter the year which should be greater than 2017 i.e above test set values

step_yr = 2017

yr = int(input('Enter the Year to Predict:'))

c = yr - step_yr

e = c-1

b = pred[120+(e*12) : 120+(e*12)+12].mean(axis=0)
print(b)

if b >= 0.5 and b <= 0.9:

    print(yr, 'is Weak El-Nino')

elif b >= 1.0 and b <= 1.4:

    print('It is Moderate El-Nino')

elif b >= 1.5 and b <= 1.9:

    print(yr, 'is Strong El-Nino')

elif b >= 2:

    print(yr, 'is Very Strong El-Nino')

elif b <=-0.5 and b >= -0.9:

    print(yr, 'is Weak La-Nina')

elif b <= -1 and b >= -1.4:

    print(yr, 'is Moderate La-Nina')

elif b <= -1.5:

    print(yr, 'is Strong La-Nina')

else:

    print(yr, 'is a Moderate Year')
# Now plot the graph of future predicted values for that generate a date range series upto 2027

dates1 = pd.date_range(start = '2008-01', freq = 'MS', end = '2027-12')

dates1
new_df = pd.DataFrame({'Predicted_values':pred[:,0]})

new_df.set_index(dates1, inplace = True)
new_df.head()
# Now plot the graph to see how over train, test and future values are predicted

plt.figure(figsize=(15,5))

plt.plot(train['Mean'])

plt.plot(test['Mean'], color = 'black')

plt.plot(test['Predictions'], color = 'orange')

plt.plot(new_df['Predicted_values'][120:], color = 'red')

plt.xlabel('Years')

plt.ylabel('Mean')

plt.legend(loc = True)

plt.title('Forecasting on Actual data')
