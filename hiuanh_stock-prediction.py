import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 20,10

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
import math
from sklearn.metrics import mean_squared_error
import seaborn as sns

train = pd.read_csv('../input/data-for-testing/TRAIN.csv')
test = pd.read_csv('../input/data-for-testing/TEST.csv')
valid = pd.read_csv('../input/data-for-testing/VALID.csv')
train.head()
#importing required libraries
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM

train = train.sort_index(ascending=False, axis=0)
test = test.sort_index(ascending=False, axis=0)
valid = valid.sort_index(ascending=False, axis=0)
 
train
#train['Date'] = pd.to_datetime(train.Date,format='%Y-%m-%d')
#test['Date'] = pd.to_datetime(test.Date,format='%Y-%m-%d')
#train.index = train['Date']
#test.index = test['Date']

#plot
plt.plot(train['Close'])
plt.plot(test[['Close']])
plt.plot(valid['Close'])
target="Close"

x_train=train.drop([target,"Date"],axis=1)
y_train=train[target]
x_test=test.drop([target,"Date"],axis=1)
y_test=test[target]

x_valid=valid.drop([target,"Date"],axis=1)
y_valid=valid[target]

x_train.shape, x_test.shape, x_valid.shape, y_train.shape, y_test.shape, y_valid.shape
scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
x_train = scaler.transform(x_train)
x_valid = scaler.transform(x_valid)
x_test
x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))
x_test = np.reshape(x_test, (x_test.shape[0],x_test.shape[1],1))
x_valid = np.reshape(x_test, (x_valid.shape[0],x_valid.shape[1],1))
# LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1],1)))
model.add(LSTM(units=50))
model.add(Dense(1))

model.compile(loss='mean_absolute_error', optimizer='adam')
model.fit(x_train, y_train, epochs=500, batch_size=4, verbose=2)

trainPredict = model.predict(x_train)
testPredict = model.predict(x_test)
validPredict = model.predict(x_valid)


scoresTrain = model.evaluate(x_train, y_train, verbose=2)
print("Train: ",  scoresTrain * 100)
scoresTest = model.evaluate(x_test, y_test, verbose=2)
print("Test: ",  scoresTest * 100) 
scoresValid = model.evaluate(x_valid, y_valid, verbose=2)
print("Valid: ",  scoresValid * 100) 
y_test.shape
scaler.fit_transform(trainPredict)
trainPredict = scaler.transform(trainPredict)
testPredict = scaler.transform(testPredict)
validPredict = scaler.transform(validPredict)

y_train = scaler.transform([y_train])
y_test = scaler.transform([y_test])
y_valid = scaler.transform([y_valid])
train
testPredict = scaler.inverse_transform(testPredict)
trainPredict = scaler.inverse_transform(trainPredict)
validPredict = scaler.inverse_transform(validPredict)
# Plot
test['Predictions'] = testPredict
valid['Predictions'] = validPredict
train['Predictions'] = trainPredict
#plt.plot(train['Close']) # Xanh dương
#plt.plot(test[['Close','Predictions']]) #  Cam,  xanh lá


#plt.plot(train[['Close','Predictions']])
#plt.plot(test[['Close','Predictions']])
plt.plot(valid[['Close','Predictions']]) 
