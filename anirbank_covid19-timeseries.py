import pandas
import matplotlib.pyplot as plt
import numpy
import pandas as pd
import math
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import PolynomialFeatures
from pandas.plotting import scatter_matrix

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import SGDRegressor
from keras.models import Sequential
from keras.layers import Dense   
from keras import optimizers

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import numpy as np
from pandas import DataFrame
from pandas import concat
from keras.models import load_model
from keras import optimizers
from matplotlib import pyplot
from math import sqrt
from keras import optimizers

# Any results you write to the current directory are saved as output.
import tensorflow as tf
print(tf.__version__)
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
np.random.seed(7)

# load the dataset
dataframe = pandas.read_csv('/kaggle/input/us-counties-covid-19-dataset/us-counties.csv')
dataframe.head()
dataframe['pd_date'] = pd.to_datetime(dataframe.date)
dataframe=dataframe.sort_values(by='pd_date',ascending=True)
print(dataframe['pd_date'].max())
print(dataframe['pd_date'].min())
dataframe.state.value_counts()[:20]
dataframe_Tx = dataframe[dataframe.state == 'Texas']
len(dataframe_Tx)
dataframe_Tx.county.value_counts()
#dataframe_wton = dataframe[(dataframe.county=='Washington') & (dataframe.state=='Oregon')]
dataframe_wton = dataframe[(dataframe.county=='El Paso') & (dataframe.state=='Texas')]
len(dataframe_wton)
# Filter out only the cases and deaths values
#dataframe_wton.index=dataframe_wton['pd_date']
dataframe_wton = dataframe_wton.iloc[:,4:6]
dataframe_wton.tail(20)
dataset_cases = dataframe_wton.values[:,0:1]
dataset_cases = dataset_cases.astype('float32')
dataset_deaths = dataframe_wton.values[:,1:2]
dataset_deaths = dataset_deaths.astype('float32')
plt.title("Number of COVID 19 cases by day for El Paso County")
plt.plot(dataset_cases)
plt.show()
plt.title("Number of COVID 19 deaths by day for El Paso County")
plt.plot(dataset_deaths)
plt.show()
dataset = dataset_cases
len(dataset)
train_size = int(len(dataset)) - 29
#test_size = 14
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
print(len(train), len(test))
#create dataframe series for t+1,t+2,t+3, to be used as y values, during Supervised Learning
#lookback = 5, means 5 values of TimeSeries (x) are used to predict the value at time t+1,t+2,t+3 (y)
def createSupervisedTrainingSet(dataset,lookback):

    df = DataFrame()
    x = dataset
    
    len_series = x.shape[0]

    df['t'] = [x[i] for i in range(x.shape[0])]
    #create x values at time t
    x=df['t'].values
    
    cols=list()
  
    df['t+1'] = df['t'].shift(-lookback)
    cols.append(df['t+1'])
    df['t+2'] = df['t'].shift(-(lookback+1))
    cols.append(df['t+2'])
    df['t+3'] = df['t'].shift(-(lookback+2))
    cols.append(df['t+3'])
    agg = concat(cols,axis=1)
    y=agg.values

    x = x.reshape(x.shape[0],1)

    len_X = len_series-lookback-2
    X=np.zeros((len_X,lookback,1))
    Y=np.zeros((len_X,3))
 
    for i in range(len_X):
        X[i] = x[i:i+lookback]
        Y[i] = y[i]

    return X,Y


look_back = 3
trainX, trainY = createSupervisedTrainingSet(train, look_back)
testX,testY = createSupervisedTrainingSet(test, look_back)
testY=testY.reshape(testY.shape[0],testY.shape[1])
trainY=trainY.reshape(trainY.shape[0],trainY.shape[1])
print(trainX.shape)
print(trainY.shape)
print(testX.shape)
print(testY.shape)
#Check the sample train X and train Y, and match with original time series data
print1 = trainY[13,:].reshape(1,-1)
print("Train X at index 13")
print(np.around((trainX[13,:,:])))
print("Train Y at index 13")
print(np.around((print1)))
print("Actual Data")
print(np.around((dataset[13:19])))        
#We used a lookback value of 5
#We inspect the X,Y values at a random index: 13
#As can be seen the 5 values of Time Series (Call Volume) from index 13 are being used as X to 
#predict the 3 values coming next (t+1,t+2,t+3)
model = Sequential()
model.add(LSTM(16,activation='relu',return_sequences=True, input_shape=(look_back, 1)))
model.add(LSTM(8, activation='relu'))
model.add(Dense(3))
myOptimizer = optimizers.Adam(lr=0.1, beta_1=0.9, beta_2=0.999, epsilon=0.01, decay=0.0)
model.compile(loss='mean_squared_error', optimizer=myOptimizer)
history = model.fit(trainX, trainY, epochs=200,  validation_data=(testX,testY), batch_size=5, verbose=2)
odel = tf.keras.models.Sequential([
  tf.keras.layers.Conv1D(filters=32, kernel_size=5,
                      strides=1, padding="causal",
                      activation="relu",
                      input_shape=[None, 1]),
  tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, return_sequences=True)),
  tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, return_sequences=True)),
  tf.keras.layers.Dense(1),
  tf.keras.layers.Lambda(lambda x: x * 200)
])
lr_schedule = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 1e-8 * 10**(epoch / 20))
optimizer = tf.keras.optimizers.SGD(lr=1e-8, momentum=0.9)
model.compile(loss=tf.keras.losses.Huber(),
              optimizer=optimizer,
              metrics=["mae"])
history = model.fit(trainX, trainY, epochs=100, callbacks=[lr_schedule])
plt.semilogx(history.history["lr"], history.history["loss"])
plt.axis([1e-8, 1e-4, 0, 30])

model = tf.keras.models.Sequential([
  tf.keras.layers.Conv1D(filters=32, kernel_size=3,
                      strides=1, padding="VALID",
                      activation="relu",
                      input_shape=[None, 1]),
  tf.keras.layers.LSTM(32, return_sequences=True),
  tf.keras.layers.LSTM(32, return_sequences=True),
  tf.keras.layers.Dense(1)
])

optimizer = tf.keras.optimizers.SGD(lr=1e-1, momentum=0.9)
model.compile(loss=tf.keras.losses.Huber(),
              optimizer=optimizer,
              metrics=["mae"])
history = model.fit(trainX, trainY, epochs=200,  validation_data=(testX,testY), batch_size=5, verbose=2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'], color=  'red')
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
#Once the model is trained, use it to make a prediction on the test data
testPredict = model.predict(testX)
predictUnscaled = np.around((testPredict))
testYUnscaled = np.around((testY))
#print the actual and predicted values at t+3
print("Actual values of COVID 19 cases")
print(testYUnscaled[:,0])
print("Predicted values of COVID 19 cases")
print(predictUnscaled[:,0])
pyplot.plot(testPredict[:,0], color='red')
pyplot.plot(testY[:,0])
pyplot.legend(['Predicted','Actual'])
pyplot.title('Actual vs Predicted at time t+1')
pyplot.show()
#Evaluate the RMSE values at t+1,t+2,t+3 to compare with other approaches, and select the best approach
def evaluate_forecasts(actuals, forecasts, n_seq):
    	for i in range(n_seq):
            actual = actuals[:,i]
            predicted = forecasts[:,i]
            rmse = sqrt(mean_squared_error(actual, predicted))
            print('t+%d RMSE: %f' % ((i+1), rmse))
        
evaluate_forecasts(testYUnscaled, predictUnscaled,3)
dataset_cases[:,0].astype(int)
from statsmodels.tsa.stattools import adfuller
from numpy import log
result = adfuller(dataframe_wton['cases'])
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
plt.rcParams.update({'figure.figsize':(9,7), 'figure.dpi':120})

# Original Series
fig, axes = plt.subplots(3, 2, sharex=True)
axes[0, 0].plot(dataframe_wton['cases']); axes[0, 0].set_title('Original Series')
plot_acf(dataframe_wton['cases'], ax=axes[0, 1])

# 1st Differencing
axes[1, 0].plot(dataframe_wton['cases'].diff()); axes[1, 0].set_title('1st Order Differencing')
plot_acf(dataframe_wton['cases'].diff().dropna(), ax=axes[1, 1])

# 2nd Differencing
axes[2, 0].plot(dataframe_wton['cases'].diff().diff()); axes[2, 0].set_title('2nd Order Differencing')
plot_acf(dataframe_wton['cases'].diff().diff().dropna(), ax=axes[2, 1])
# PACF plot of 1st differenced series
plt.rcParams.update({'figure.figsize':(9,3), 'figure.dpi':120})

fig, axes = plt.subplots(1, 2, sharex=True)
axes[0].plot(dataframe_wton['cases'].diff()); axes[0].set_title('1st Differencing')
axes[1].set(ylim=(0,5))
plot_pacf(dataframe_wton['cases'].diff().dropna(), ax=axes[1])

plt.show()
plt.rcParams.update({'figure.figsize':(9,3), 'figure.dpi':120})

fig, axes = plt.subplots(1, 2, sharex=True)
axes[0].plot(dataframe_wton['cases'].diff()); axes[0].set_title('1st Differencing')
axes[1].set(ylim=(0,1.2))
plot_acf(dataframe_wton['cases'].diff().dropna(), ax=axes[1])

plt.show()
dataframe_wton = dataframe[(dataframe.county=='El Paso') & (dataframe.state=='Texas')]
dataframe_wton.index=dataframe_wton['pd_date']
dataframe_wton = dataframe_wton.iloc[:,4:6]
test_range = pd.to_datetime(dataframe_wton.index[160:])
from statsmodels.tsa.arima_model import ARIMA

# 1,1,2 ARIMA Model
model = ARIMA(dataframe_wton['cases'].astype(float), order=(1,1,2))
model_fit = model.fit(disp=0)
print(model_fit.summary())
# Plot residual errors
residuals = pd.DataFrame(model_fit.resid)
fig, ax = plt.subplots(1,2)
residuals.plot(title="Residuals", ax=ax[0])
residuals.plot(kind='kde', title='Density', ax=ax[1])
plt.show()

# Actual vs Fitted
trange = np.arange(160,189)
trange
model_fit.plot_predict(dynamic=False)
plt.show()
train = dataframe_wton.iloc[0:160, :]
test = dataframe_wton.iloc[160:, :]

arima = ARIMA(train['cases'].astype(float), order = (1,1,2)).fit(disp = 0)

prediction = arima.plot_predict(test.index[0], test.index[-1], dynamic = True)