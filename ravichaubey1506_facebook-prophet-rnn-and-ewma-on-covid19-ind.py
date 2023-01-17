import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
sns.set_style('whitegrid')
df = pd.read_csv('../input/novel-corona-virus-2019-dataset/covid_19_data.csv',index_col='ObservationDate',parse_dates=True)
covidIndia = df[df['Country/Region'] == 'India']
covidIndia.drop(['SNo','Last Update','Province/State'],axis=1,inplace = True)
covidIndia.head()
covidIndia.tail()
print("Shape of Data is ==> ",covidIndia.shape)
covidIndia.index[:10]
covidIndia.index.freq = 'D'
covidIndia.index[:10]
from statsmodels.tsa.seasonal import seasonal_decompose

result = seasonal_decompose(covidIndia['Confirmed'], model='mul')

from pylab import rcParams
rcParams['figure.figsize'] = 12,8
result.plot();
train_data = covidIndia.iloc[:78]
test_data = covidIndia.iloc[78:]

from statsmodels.tsa.holtwinters import ExponentialSmoothing

fitted_model = ExponentialSmoothing(train_data['Confirmed'],trend='mul').fit()

test_predictions = fitted_model.forecast(15).rename('Confirmed Forecast')
print("Prediction ==> \n",test_predictions[:5])
print("\n","Actual Data ==> \n",test_data[:5]['Confirmed'])
fig = plt.figure(dpi = 120)
ax = plt.axes()
ax.set(xlabel = 'Date',ylabel = 'Count of Cases',title = 'Comparision : Test VS Prediction')
train_data['Confirmed'].plot(legend=True,label='TRAIN',lw = 2)
test_data['Confirmed'].plot(legend=True,label='TEST',figsize=(8,4),lw = 2)
test_predictions.plot(legend=True,label='PREDICTION',lw = 2);
fig = plt.figure(dpi = 120)
ax = plt.axes()
ax.set(xlabel = 'Date',ylabel = 'Count of Cases',title = 'Comparision : Test VS Prediction (Zoon In)')
test_data['Confirmed'].plot(legend=True,label='TEST DATA',figsize=(8,4),lw = 2)
test_predictions.plot(legend=True,label='PREDICTION',xlim=['2020-04-23','2020-04-27'],lw = 2);
from sklearn.metrics import mean_squared_error,mean_absolute_error

print("MAE ==> ",mean_absolute_error(test_data['Confirmed'],test_predictions))
print("MSE ==> ",mean_squared_error(test_data['Confirmed'],test_predictions))
print("RMSE ==> ",np.sqrt(mean_squared_error(test_data['Confirmed'],test_predictions)))
test_data.describe()['Confirmed']['std']
final_model = ExponentialSmoothing(train_data['Confirmed'],trend='mul').fit()
forecast_predictions = final_model.forecast(30)
fig = plt.figure(dpi = 120)
ax = plt.axes()
ax.set(xlabel = 'Date',ylabel = 'Count of Cases',title = 'Forecast : (May 1, 2020) to (May 15, 2020)')
covidIndia['Confirmed'].plot(figsize=(8,4),lw = 2,legend = True,label = 'Actual Confirmed')
forecast_predictions.plot(lw=2,legend = True,label = 'Forecast Confirmed',xlim = ['2020-04-20','2020-05-15']);
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

train = pd.DataFrame(covidIndia.iloc[:78,1])
test = pd.DataFrame(covidIndia.iloc[78:,1])

scaler.fit(train)
scaled_train = scaler.transform(train)
scaled_test = scaler.transform(test)
print("Scaled Train Set ==> \n", scaled_train[:5],"\n")
print("Scaled Test Set==> \n", scaled_test[:5])
from keras.preprocessing.sequence import TimeseriesGenerator
n_input = 15
n_features = 1
generator = TimeseriesGenerator(scaled_train, scaled_train, length=n_input, batch_size=1)
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

# define model
model = Sequential()
model.add(LSTM(150, activation='relu', input_shape=(n_input, n_features)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
model.summary()
# fit model
model.fit_generator(generator,epochs=25)
loss_per_epoch = model.history.history['loss']
fig = plt.figure(dpi = 120,figsize = (8,4))
ax = plt.axes()
ax.set(xlabel = 'Number of Epochs',ylabel = 'MSE Loss',title = 'Loss Curve of RNN LSTM')
plt.plot(range(len(loss_per_epoch)),loss_per_epoch,lw = 2);
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
true_predictions = scaler.inverse_transform(test_predictions)
test['Predictions'] = true_predictions
test.head()
fig = plt.figure(dpi = 120)
ax=plt.axes()
test.plot(legend=True,figsize=(14,6),lw = 2,ax=ax)
plt.xlabel('Date')
plt.ylabel('Count of Cases')
plt.title('Comparision B/W Test and Prediction')
plt.show();
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

train = pd.DataFrame(covidIndia.iloc[:,1])


scaler.fit(train)
scaled_train = scaler.transform(train)

n_input = 15
n_features = 1
generator = TimeseriesGenerator(scaled_train, scaled_train, length=n_input, batch_size=1)

# define model
model = Sequential()
model.add(LSTM(150, activation='relu', input_shape=(n_input, n_features)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

# fit model
model.fit_generator(generator,epochs=25)
loss_per_epoch = model.history.history['loss']
fig = plt.figure(dpi = 120,figsize = (8,4))
ax = plt.axes()
ax.set(xlabel = 'Number of Epochs',ylabel = 'MSE Loss',title = 'Loss Curve of RNN LSTM')
plt.plot(range(len(loss_per_epoch)),loss_per_epoch,lw = 2);
forecast = []

first_eval_batch = scaled_train[-n_input:]
current_batch = first_eval_batch.reshape((1, n_input, n_features))

for i in range(15):
    current_pred = model.predict(current_batch)[0]
    forecast.append(current_pred) 
    current_batch = np.append(current_batch[:,1:,:],[[current_pred]],axis=1)

forecast= scaler.inverse_transform(forecast)
forecast = pd.DataFrame({'Forecast':forecast.flatten()})
forecast.index = np.arange('2020-05-01',15,dtype='datetime64[D]')
forecast.head()
fig = plt.figure(dpi=120,figsize = (14,6))
ax = plt.axes()
ax.set(xlabel = 'Date',ylabel = 'Count of Cases (Lacs)',title = 'Forecast : (May 1, 2020) to (May 15, 2020)')
forecast.plot(label = 'Forecast',ax=ax,color='red',lw=2);
df = pd.DataFrame(covidIndia.iloc[:,1])
df.reset_index(inplace = True)
df.head()
df.columns = ['ds','y']
df.head()
fig = plt.figure(dpi = 120)
axes = plt.axes()
axes.set(xlabel = 'Date',ylabel = 'Count of Cases',title = 'Trend')
df.plot(x='ds',y='y',figsize=(8,4),lw=2,color = 'blue',ax=axes);
train = df.iloc[:78]
test = df.iloc[78:]
from fbprophet import Prophet
m = Prophet()
m.fit(train)
future = m.make_future_dataframe(periods=15)
forecast = m.predict(future)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(5)
test.tail(5)
fig = plt.figure(dpi = 120)
ax = plt.axes()
ax.set(xlabel = 'Date',ylabel = 'Count of Cases',title = 'Comparision B/W Test & Prediction')
forecast.plot(x='ds',y='yhat',label='Predictions',legend=True,figsize=(8,4),ax=ax,lw=2)
test.plot(x='ds',y='y',label='True Miles',legend=True,ax=ax,xlim=('2020-04-17','2020-05-01'),lw=2);
from statsmodels.tools.eval_measures import rmse
predictions = forecast.iloc[-15:]['yhat']
print("RMSE ==> ",rmse(predictions,test['y']))
print("Test Mean ==> ",test.mean())
from fbprophet import Prophet
m = Prophet()
m.fit(df)
future = m.make_future_dataframe(periods=15)
forecast = m.predict(future)
df.tail()
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(5)
fig = plt.figure(dpi = 120 )
axes = plt.axes()
m.plot(forecast, figsize = (8,4),ax=axes)
plt.xlabel('Date')
plt.ylabel('Count of Cases')
plt.title('Forecast')
plt.xticks(rotation = 90);
plt.figure(dpi = 120)
axes = plt.axes()
m.plot(forecast,ax=axes,figsize = (8,4))
start = pd.to_datetime(['2020-04-25'])
end = pd.to_datetime(['2020-05-15'])
plt.xlabel('Date')
plt.ylabel('Count of Cases')
plt.title('Forecast : (May 1, 2020) - (May 15, 2020)')
plt.xlim(start,end)
plt.ylim(20000,60000)
plt.xticks(rotation = 90);