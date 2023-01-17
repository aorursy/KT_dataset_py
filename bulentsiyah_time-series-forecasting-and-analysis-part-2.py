import pandas as pd

import numpy as np

%matplotlib inline

import matplotlib.pyplot as plt
df = pd.read_csv('/kaggle/input/for-simple-exercises-time-series-forecasting/Alcohol_Sales.csv',index_col='DATE',parse_dates=True)

df.index.freq = 'MS'

df.head()
df.columns = ['Sales']

df.plot(figsize=(12,8))
from statsmodels.tsa.seasonal import seasonal_decompose



results = seasonal_decompose(df['Sales'])

results.observed.plot(figsize=(12,2))
results.trend.plot(figsize=(12,2))
results.seasonal.plot(figsize=(12,2))
results.resid.plot(figsize=(12,2))
print("len(df)", len(df))



train = df.iloc[:313]

test = df.iloc[313:]





print("len(train)", len(train))

print("len(test)", len(test))
from sklearn.preprocessing import MinMaxScaler



scaler = MinMaxScaler()



# IGNORE WARNING ITS JUST CONVERTING TO FLOATS

# WE ONLY FIT TO TRAININ DATA, OTHERWISE WE ARE CHEATING ASSUMING INFO ABOUT TEST SET

scaler.fit(train)

scaled_train = scaler.transform(train)

scaled_test = scaler.transform(test)
from keras.preprocessing.sequence import TimeseriesGenerator

scaled_train[0]
# define generator

n_input = 2

n_features = 1

generator = TimeseriesGenerator(scaled_train, scaled_train, length=n_input, batch_size=1)



print('len(scaled_train)',len(scaled_train))

print('len(generator)',len(generator))  # n_input = 2
# What does the first batch look like?

X,y = generator[0]



print(f'Given the Array: \n{X.flatten()}')

print(f'Predict this y: \n {y}')
# Let's redefine to get 12 months back and then predict the next month out

n_input = 12

generator = TimeseriesGenerator(scaled_train, scaled_train, length=n_input, batch_size=1)



# What does the first batch look like?

X,y = generator[0]



print(f'Given the Array: \n{X.flatten()}')

print(f'Predict this y: \n {y}')
from keras.models import Sequential

from keras.layers import Dense

from keras.layers import LSTM



# define model

model = Sequential()

model.add(LSTM(100, activation='relu', input_shape=(n_input, n_features)))

model.add(Dense(1))

model.compile(optimizer='adam', loss='mse')

model.summary()
# fit model

model.fit_generator(generator,epochs=50)
model.history.history.keys()

loss_per_epoch = model.history.history['loss']

plt.plot(range(len(loss_per_epoch)),loss_per_epoch)
first_eval_batch = scaled_train[-12:]

first_eval_batch
first_eval_batch = first_eval_batch.reshape((1, n_input, n_features))

model.predict(first_eval_batch)
scaled_test[0]
test_predictions = []



first_eval_batch = scaled_train[-n_input:]

current_batch = first_eval_batch.reshape((1, n_input, n_features))



for i in range(len(test)):

    

    # get prediction 1 time stamp ahead ([0] is for grabbing just the number instead of [array])

    current_pred = model.predict(current_batch)[0]

    

    # store prediction

    test_predictions.append(current_pred) 

    

    # update batch to now include prediction and drop first value

    current_batch = np.append(current_batch[:,1:,:],[[current_pred]],axis=1)

    

test_predictions
scaled_test
true_predictions = scaler.inverse_transform(test_predictions)

true_predictions
test['Predictions'] = true_predictions

test
test.plot(figsize=(12,8))
model.save('my_rnn_model.h5')

'''from keras.models import load_model

new_model = load_model('my_rnn_model.h5')'''
import pandas as pd

import numpy as np

%matplotlib inline

import matplotlib.pyplot as plt



df = pd.read_csv('../input/for-simple-exercises-time-series-forecasting/energydata_complete.csv',index_col='date', infer_datetime_format=True)

df.head()
df.info()
df['Windspeed'].plot(figsize=(12,8))
df['Appliances'].plot(figsize=(12,8))
df = df.loc['2016-05-01':]

df = df.round(2)



print('len(df)',len(df))

test_days = 2

test_ind = test_days*144 # 24*60/10 = 144

test_ind
train = df.iloc[:-test_ind]

test = df.iloc[-test_ind:]
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()



# IGNORE WARNING ITS JUST CONVERTING TO FLOATS

# WE ONLY FIT TO TRAININ DATA, OTHERWISE WE ARE CHEATING ASSUMING INFO ABOUT TEST SET

scaler.fit(train)



scaled_train = scaler.transform(train)

scaled_test = scaler.transform(test)
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator



# define generator

length = 144 # Length of the output sequences (in number of timesteps)

batch_size = 1 #Number of timeseries samples in each batch

generator = TimeseriesGenerator(scaled_train, scaled_train, length=length, batch_size=batch_size)
print('len(scaled_train)',len(scaled_train))

print('len(generator) ',len(generator))



X,y = generator[0]



print(f'Given the Array: \n{X.flatten()}')

print(f'Predict this y: \n {y}')
from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense,LSTM



scaled_train.shape
# define model

model = Sequential()



# Simple RNN layer

model.add(LSTM(100,input_shape=(length,scaled_train.shape[1])))



# Final Prediction (one neuron per feature)

model.add(Dense(scaled_train.shape[1]))



model.compile(optimizer='adam', loss='mse')



model.summary()
from tensorflow.keras.callbacks import EarlyStopping

early_stop = EarlyStopping(monitor='val_loss',patience=1)

validation_generator = TimeseriesGenerator(scaled_test,scaled_test, 

                                           length=length, batch_size=batch_size)



model.fit_generator(generator,epochs=10,

                    validation_data=validation_generator,

                   callbacks=[early_stop])
model.history.history.keys()



losses = pd.DataFrame(model.history.history)

losses.plot()
first_eval_batch = scaled_train[-length:]

first_eval_batch
first_eval_batch = first_eval_batch.reshape((1, length, scaled_train.shape[1]))

model.predict(first_eval_batch)
scaled_test[0]
n_features = scaled_train.shape[1]

test_predictions = []



first_eval_batch = scaled_train[-length:]

current_batch = first_eval_batch.reshape((1, length, n_features))



for i in range(len(test)):

    

    # get prediction 1 time stamp ahead ([0] is for grabbing just the number instead of [array])

    current_pred = model.predict(current_batch)[0]

    

    # store prediction

    test_predictions.append(current_pred) 

    

    # update batch to now include prediction and drop first value

    current_batch = np.append(current_batch[:,1:,:],[[current_pred]],axis=1)

    
true_predictions = scaler.inverse_transform(test_predictions)



true_predictions = pd.DataFrame(data=true_predictions,columns=test.columns)

true_predictions
import pandas as pd

from fbprophet import Prophet
df = pd.read_csv('../input/for-simple-exercises-time-series-forecasting/Miles_Traveled.csv')

df.head()
df.columns = ['ds','y']

df['ds'] = pd.to_datetime(df['ds'])

df.info()
pd.plotting.register_matplotlib_converters()



try:

    df.plot(x='ds',y='y',figsize=(18,6))

except TypeError as e:

    figure_or_exception = str("TypeError: " + str(e))

else:

    figure_or_exception = df.set_index('ds').y.plot().get_figure()

print('len(df)',len(df))

print('len(df) - 12 = ',len(df) - 12)
train = df.iloc[:576]

test = df.iloc[576:]
# This is fitting on all the data (no train test split in this example)

m = Prophet()

m.fit(train)
future = m.make_future_dataframe(periods=12,freq='MS')

forecast = m.predict(future)
forecast.tail()
test.tail()
forecast.columns
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(12)
m.plot(forecast);
import matplotlib.pyplot as plt

%matplotlib inline

m.plot(forecast)

plt.xlim(pd.to_datetime('2003-01-01'),pd.to_datetime('2007-01-01'))
m.plot_components(forecast);
from statsmodels.tools.eval_measures import rmse

predictions = forecast.iloc[-12:]['yhat']

predictions
test['y']
rmse(predictions,test['y'])
test.mean()
from fbprophet.diagnostics import cross_validation,performance_metrics

from fbprophet.plot import plot_cross_validation_metric



len(df)

len(df)/12



# Initial 5 years training period

initial = 5 * 365

initial = str(initial) + ' days'

# Fold every 5 years

period = 5 * 365

period = str(period) + ' days'

# Forecast 1 year into the future

horizon = 365

horizon = str(horizon) + ' days'



df_cv = cross_validation(m, initial=initial, period=period, horizon = horizon)



df_cv.head()
df_cv.tail()
performance_metrics(df_cv)
plot_cross_validation_metric(df_cv, metric='rmse');
plot_cross_validation_metric(df_cv, metric='mape');