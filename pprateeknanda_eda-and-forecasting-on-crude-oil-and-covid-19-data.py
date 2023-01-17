import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd 
from matplotlib import rcParams
%matplotlib inline
import plotly.express as px
import datetime
from fbprophet import diagnostics
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from fbprophet.diagnostics import performance_metrics
pd.set_option('display.max_columns', None)
import os
import itertools
import statsmodels.api as sm
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
import math
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings("ignore")
pd.options.display.float_format = '{:.0f}'.format
import math
from zipfile import ZipFile
import statsmodels.api as sm
from statsmodels.tsa.ar_model import AutoReg
from pandas.plotting import lag_plot
from pandas.plotting import autocorrelation_plot
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.arima_model import ARMA

oil_price = pd.read_csv(r'C:\Users\205694\Desktop\Sentiment Analysis\Crude Oil\ntt-data-global-ai-challenge-06-2020\Crude_oil_trend_From1986-01-02_To2020-06-15.csv')
cases=pd.read_csv(r'C:\Users\205694\Desktop\Sentiment Analysis\Crude Oil\ntt-data-global-ai-challenge-06-2020\COVID-19_and_Price_dataset.csv')
oil_price_v = oil_price.copy()
oil_price_v['Date']=pd.to_datetime(oil_price_v['Date'])
oil_price_v = oil_price_v.sort_values('Date')
oil_price_v.set_index('Date', inplace=True)
oil_price_v=oil_price_v.loc[datetime.date(year=1986,month=1,day=1):]
plt.figure(figsize=(15,8))
plt.plot(oil_price_v,color = 'tab:blue')
plt.title('Oil Price Fluctuation since 1986 to Present')
plt.xlabel('Year')
plt.ylabel('Oil Price ')
plt.show()
fig, axis1 = plt.subplots(figsize = (25,15))
plt.grid()
plt.xticks(rotation="vertical")
line1, = axis1.plot(cases['Date'], cases['Price'], color = 'tab:red')
axis1.set_ylabel("Oil price")
axis1.set_xlabel("Date")
axis2 = axis1.twinx()
line2, = axis2.plot(cases['Date'], cases['World_total_cases'], color = 'tab:blue')
axis2.set_ylabel("Number of cases")
fig.legend((line1, line2),('Oil price','Total cases'),'upper right')
plt.title("Daily World Cases vs. Oil Price between 31-12-2019 to 15-06-2020")
plt.show()
fig, axis1 = plt.subplots(figsize = (25,15))
plt.grid()
plt.xticks(rotation="vertical")
line1, = axis1.plot(cases['Date'], cases['Price'], color = 'tab:red')
axis1.set_ylabel("Oil price")
axis1.set_xlabel("Date")
axis2 = axis1.twinx()
line2, = axis2.plot(cases['Date'], cases['World_total_cases'],color = 'tab:blue')
axis2.set_ylabel("Number of cases")
axis3 = axis1.twinx()
line3, = axis3.plot(cases['Date'], cases['World_total_deaths'],linestyle='dashed',color = 'tab:green')
fig.legend((line1, line2, line3),('Oil price','Total cases','Total deaths'),'upper right')
plt.title("Daily World Total Cases vs. Daily World Total Death vs. Oil price between 31-12-2019 to 15-06-2020")
plt.show()
fig,axis = plt.subplots(figsize=(25,15))
plt.grid()
plt.xticks(rotation='vertical')
axis.plot(cases['Date'],cases['World_new_cases'])
axis.set(xlabel="Date",ylabel="World New Cases",title="World New Cases between 31-12-2019 to 15-06-2020")
plt.show()
fig,axis = plt.subplots(figsize=(25,15))
plt.grid()
plt.xticks(rotation='vertical')
axis.plot(cases['Date'],cases['World_total_deaths'])
axis.set(xlabel="Date",ylabel="World Total Deaths",title="World Total Deaths between 31-12-2019 to 15-06-2020")
plt.show()
fig, axis = plt.subplots(figsize=(25,15))
plt.grid()
plt.xticks(rotation='vertical')
axis.plot(cases['Date'],cases['World_new_deaths'])
axis.set(xlabel="Date",ylabel="World New Deaths",title="World New Deaths between 31-12-2019 to 15-06-2020")
plt.show()
fig, axis1 = plt.subplots(figsize = (25,15))
plt.grid()
plt.xticks(rotation="vertical")
line1, = axis1.plot(cases['Date'], cases['Price'], color = 'tab:red')
axis1.set_ylabel("Oil price")
axis1.set_ylabel("Date")
axis2 = axis1.twinx()
line2, = axis2.plot(cases['Date'], cases['UnitedStates_total_cases'],color = 'tab:blue')
axis2.set_ylabel("Number of cases")
fig.legend((line1, line2),('Oil price', 'Total cases'),'upper right')
plt.title("Daily USA cases vs. Oil price between 31-12-2019 to 15-06-2020")
plt.show()
fig, axis1 = plt.subplots(figsize = (25,15))
plt.grid()
plt.xticks(rotation="vertical")
line1, = axis1.plot(cases['Date'], cases['Price'], color = 'tab:red')
axis1.set_ylabel("Oil price")
axis1.set_xlabel("Date")
axis2 = axis1.twinx()
line2, = axis2.plot(cases['Date'], cases['Italy_total_cases'],color = 'tab:blue')
axis2.set_ylabel("Number of cases")
fig.legend((line1, line2),('Oil price', 'Total cases'),'upper right')
plt.title("Daily Italy cases vs. Oil price between 31-12-2019 to 15-06-2020")
plt.show()
fig, axis1 = plt.subplots(figsize = (25,15))
plt.grid()
plt.xticks(rotation="vertical")
line1, = axis1.plot(cases['Date'], cases['Price'], color = 'tab:red')
axis1.set_ylabel("Oil price")
axis1.set_xlabel("Date")
axis2 = axis1.twinx()
line2, = axis2.plot(cases['Date'], cases['China_total_cases'],color = 'tab:blue')
axis2.set_ylabel("Number of cases")
fig.legend((line1, line2),('Oil price', 'Total cases'),'upper right')
plt.title("Daily China cases vs. Oil price between 31-12-2019 to 15-06-2020")
plt.show()
fig, axis1 = plt.subplots(figsize = (25,15))
plt.grid()
plt.xticks(rotation="vertical")
line1, = axis1.plot(cases['Date'], cases['Price'], color = 'tab:red')
axis1.set_ylabel("Oil price")
axis1.set_xlabel("Date")
axis2 = axis1.twinx()
line2, = axis2.plot(cases['Date'], cases['France_total_cases'],color = 'tab:blue')
axis2.set_ylabel("Number of cases")
fig.legend((line1, line2),('Oil price', 'Total cases'),'upper right')
plt.title("Daily France cases vs. Oil price between 31-12-2019 to 15-06-2020")
plt.show()
oil_price_prophet=oil_price[(oil_price['Date'] >= '2016-06-17') & (oil_price['Date'] < '2020-07-06')]
oil_price_prophet.reset_index(drop=True,inplace=True)
oil_price_prophet['ds'] = pd.to_datetime(oil_price_prophet['Date']).dt.date
oil_price_prophet['y']  = oil_price_prophet['Price']
oil_price_prophet = oil_price_prophet[['ds', 'y']]
test_y = oil_price_prophet[-30:] 
oil_price_prophet = oil_price_prophet[:-30] 
test_y
from fbprophet import Prophet
from fbprophet.diagnostics import cross_validation
prophet = Prophet()
prophet.fit(oil_price_prophet)
future = prophet.make_future_dataframe(periods=90, freq = 'd')
forecast = prophet.predict(future)
# forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
future_pred = forecast[['ds','yhat']]
future_pred = future_pred.rename(columns={'ds': 'Date', 'yhat': 'Predicted Value'})
future_pred.to_csv('Prophet_Oil_crude.csv')
prophet.plot(forecast)
pred = forecast[['ds', 'yhat']][-80:]
pred['ds'] = pd.to_datetime(pred['ds']).dt.date
result = pd.merge(test_y, pred, how="inner" ,on="ds")
def mean_abs_p_error(y_true,y_pred):
    y_true, y_pred =np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true )) * 100
mape_baseline = mean_abs_p_error(result.y, result.yhat)
print('Test MAPE: %.3f' % mape_baseline)
rmse = np.sqrt(mean_squared_error(result['y'] , result['yhat'] ))
print('Test RMSE: %.3f' % rmse)
fig2 = prophet.plot_components(forecast)
from fbprophet.plot import plot_plotly
import plotly.offline as py
py.init_notebook_mode()

fig = plot_plotly(prophet, forecast) 
py.iplot(fig)
cases_world_total_prophet = cases[['Date','World_total_cases']]
cases_world_total_prophet['ds'] = pd.to_datetime(cases_world_total_prophet['Date']).dt.date
cases_world_total_prophet['y']  = cases_world_total_prophet['World_total_cases']
cases_world_total_prophet = cases_world_total_prophet[['ds', 'y']]
cases_world_total_test_y = cases_world_total_prophet[-30:] 
cases_world_total_prophet = cases_world_total_prophet[:-30] 
prophet1 = Prophet(growth='linear')
prophet1.fit(cases_world_total_prophet)
future1 = prophet1.make_future_dataframe(periods=75, freq = 'd')
forecast1 = prophet1.predict(future1)
forecast1[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
future_pred1 = forecast1[['ds','yhat']]
future_pred1 = future_pred1.rename(columns={'ds': 'Date', 'yhat': 'Predicted Value'})
future_pred1.to_csv('Prophet_World_Cases.csv')
prophet1.plot(forecast1)
pred1 = forecast1[['ds', 'yhat']][-40:]
pred1['ds'] = pd.to_datetime(pred1['ds']).dt.date
result1 = pd.merge(cases_world_total_test_y, pred1, how="inner" ,on="ds")
mape_baseline = mean_abs_p_error(result1.y, result1.yhat)
print('Test MAPE: %.3f' % mape_baseline)
rmse = np.sqrt(mean_squared_error(result1['y'] , result1['yhat'] ))
print('Test RMSE: %.3f' % rmse)
fig3 = prophet1.plot_components(forecast1)
fig = plot_plotly(prophet1, forecast1) 
py.iplot(fig)
cases_world_death_prophet = cases[['Date','World_total_deaths']]
cases_world_death_prophet['ds'] = pd.to_datetime(cases_world_death_prophet['Date']).dt.date
cases_world_death_prophet['y']  = cases_world_death_prophet['World_total_deaths']
cases_world_death_prophet = cases_world_death_prophet[['ds', 'y']]
cases_world_death_test_y = cases_world_death_prophet[-30:] 
cases_world_death_prophet = cases_world_death_prophet[:-30] 
prophet2 = Prophet(growth='linear')
prophet2.fit(cases_world_death_prophet)
future2 = prophet2.make_future_dataframe(periods=75, freq = 'd')
forecast2 = prophet2.predict(future2)
forecast2[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
future_pred2 = forecast2[['ds','yhat']]
future_pred2 = future_pred2.rename(columns={'ds': 'Date', 'yhat': 'Predicted Value'})
future_pred2.to_csv('Prophet_World_death.csv')
prophet2.plot(forecast2)
pred2 = forecast2[['ds', 'yhat']][-40:]
pred2['ds'] = pd.to_datetime(pred2['ds']).dt.date
result2 = pd.merge(cases_world_death_test_y, pred2, how="inner" ,on="ds")
mape_baseline = mean_abs_p_error(result2.y, result2.yhat)
print('Test MAPE: %.3f' % mape_baseline)
rmse = np.sqrt(mean_squared_error(result2['y'] , result2['yhat'] ))
print('Test RMSE: %.3f' % rmse)
fig2 = prophet2.plot_components(forecast2)
fig = plot_plotly(prophet2, forecast2) 
py.iplot(fig)
oil_price_lstm=oil_price[(oil_price['Date'] >= '2016-06-17') & (oil_price['Date'] < '2020-06-15')]
oil_price_lstm.reset_index(drop=True,inplace=True)
oil_price_lstm['Date']=pd.to_datetime(oil_price_lstm['Date'])
oil_price_lstm = oil_price_lstm.sort_values('Date')
oil_price_lstm.set_index('Date', inplace=True)
# oil_price_lstm=oil_price_lstm.loc[datetime.date(year=2016,month=1,day=1):]
sc = MinMaxScaler(feature_range = (0, 1))
oil_price_lstm = sc.fit_transform(oil_price_lstm)
train_data, test_data = oil_price_lstm[0:int(len(oil_price_lstm)*0.8), :], oil_price_lstm[int(len(oil_price_lstm)*0.8):len(oil_price_lstm), :]
# train_data = oil_price[(oil_price['Date'] >= '2016-06-17') & (oil_price['Date'] < '2020-04-15')]
def prepare_dataset( dataset, seq ):
    data, label =[],[]
    for i in range( len(dataset) - seq - 1 ):
        d = dataset[ i:( i+seq ), 0 ]
        data.append(d)
        label.append(dataset[i+seq,0])
    return np.array(data), np.array(label)
train_x, train_y = prepare_dataset(train_data,40)
test_x, test_y = prepare_dataset(test_data,40)
train_x = np.reshape(train_x, (train_x.shape[0], train_x.shape[1],1))
test_x = np.reshape(test_x, (test_x.shape[0], test_x.shape[1],1))
model = Sequential()
model.add(LSTM(units = 60,activation = 'relu',input_shape = (train_x.shape[1], 1)))
model.add(Dense(units = 1))
model.compile(optimizer = 'adam', loss = 'mean_squared_error')
history = model.fit(train_x, train_y, epochs = 25, batch_size = 15, validation_data = (test_x, test_y), verbose =1)
test_predict = model.predict(test_x)
test_predict = sc.inverse_transform(test_predict)
test_y = sc.inverse_transform([test_y])
y=[x for x in range(159)]
plt.figure(figsize=(8,4))
plt.plot(y, test_y[0][:159], marker='.', label="actual")
plt.plot(y, test_predict[:,0][:159], 'r', label="prediction")
plt.tight_layout()
plt.subplots_adjust(left=0.07)
plt.ylabel('Price', size=15)
plt.xlabel('Time step', size=15)
plt.legend(fontsize=15)
plt.show();
print('Test Mean Absolute Error:', mean_absolute_error(test_y[0], test_predict[:,0]))
print('Test Root Mean Squared Error:',np.sqrt(mean_squared_error(test_y[0], test_predict[:,0])))
df = oil_price_lstm.copy()
len(df['Date'])
close_data = df['Price'].values
close_data = close_data.reshape((-1,1))

split_percent = 0.80
split = int(split_percent*len(close_data))

close_train = close_data[:split]
close_test = close_data[split:]

date_train = df['Date'][:split]
date_test = df['Date'][split:]
look_back = 15
train_generator = TimeseriesGenerator(close_train, close_train, length=look_back, batch_size=20)     
test_generator = TimeseriesGenerator(close_test, close_test, length=look_back, batch_size=1)
from keras.models import Sequential
from keras.layers import LSTM, Dense

model = Sequential()
model.add(
    LSTM(10,
        activation='relu',
        input_shape=(look_back,1))
)
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

num_epochs = 25
model.fit_generator(train_generator, epochs=num_epochs, verbose=1)
prediction = model.predict_generator(test_generator)
close_train = close_train.reshape((-1))
close_test = close_test.reshape((-1))
prediction = prediction.reshape((-1))
close_data = close_data.reshape((-1))

def predict(num_prediction, model):
    prediction_list = close_data[-look_back:]
    
    for _ in range(num_prediction):
        x = prediction_list[-look_back:]
        x = x.reshape((1, look_back, 1))
        out = model.predict(x)[0][0]
        prediction_list = np.append(prediction_list, out)
    prediction_list = prediction_list[look_back-1:]
        
    return prediction_list
    
def predict_dates(num_prediction):
    last_date = df['Date'].values[-1]
    prediction_dates = pd.date_range(last_date, periods=num_prediction+1).tolist()
    return prediction_dates

num_prediction = 85
forecast = predict(num_prediction, model)
forecast_dates = predict_dates(num_prediction)
df_new = pd.DataFrame(columns=columns)
df_new['Price'] = forecast.tolist()
df_new['Date'] = forecast_dates
df_new.to_csv('LSTM_prediction.csv')
cases_world_total_lstm = cases[['Date','World_total_cases']]
# cases_world_total_lstm=cases_world_total[(cases_world_total['Date'] >= '2016-06-17') & (cases_world_total['Date'] < '2020-06-15')]
# cases_world_total_lstm.reset_index(drop=True,inplace=True)
cases_world_total_lstm['Date']=pd.to_datetime(cases_world_total_lstm['Date'])
cases_world_total_lstm = cases_world_total_lstm.sort_values('Date')
cases_world_total_lstm.set_index('Date', inplace=True)
future_test = test_x[158].T
time_length = future_test.shape[1]
future_test

sc = MinMaxScaler(feature_range = (0, 1))
cases_world_total_lstm = sc.fit_transform(cases_world_total_lstm)
train_data, test_data = cases_world_total_lstm[0:int(len(cases_world_total_lstm)*0.8), :], cases_world_total_lstm[int(len(cases_world_total_lstm)*0.8):len(cases_world_total_lstm), :]
def prepare_dataset( dataset, seq ):
    data, label =[],[]
    for i in range( len(dataset) - seq - 1 ):
        d = dataset[ i:( i+seq ), 0 ]
        data.append(d)
        label.append(dataset[i+seq,0])
    return np.array(data), np.array(label)
train_x, train_y = prepare_dataset(train_data,10)
test_x, test_y = prepare_dataset(test_data,10)
train_x = np.reshape(train_x, (train_x.shape[0], train_x.shape[1],1))
test_x = np.reshape(test_x, (test_x.shape[0], test_x.shape[1],1))
model = Sequential()
model.add(LSTM(units = 60,activation = 'relu', input_shape = (train_x.shape[1], 1)))
model.add(Dense(units = 1))
model.compile(optimizer = 'adam', loss = 'mean_squared_error')
history = model.fit(train_x, train_y, epochs = 25, batch_size = 15, validation_data = (test_x, test_y), verbose =1)
test_predict = model.predict(test_x)
test_predict = sc.inverse_transform(test_predict)
test_y = sc.inverse_transform([test_y])
print('Test Mean Absolute Error:', mean_absolute_error(test_y[0], test_predict[:,0]))
print('Test Root Mean Squared Error:',np.sqrt(mean_squared_error(test_y[0], test_predict[:,0])))
len(train_data)
cases_world_death_lstm = cases[['Date','World_total_deaths']]
# cases_world_death_lstm=cases_world_death[(cases_world_death['Date'] >= '2016-06-17') & (cases_world_death['Date'] < '2020-06-15')]
# cases_world_death_lstm.reset_index(drop=True,inplace=True)
cases_world_death_lstm['Date']=pd.to_datetime(cases_world_death_lstm['Date'])
cases_world_death_lstm = cases_world_death_lstm.sort_values('Date')
cases_world_death_lstm.set_index('Date', inplace=True)
sc = MinMaxScaler(feature_range = (0, 1))
cases_world_death_lstm = sc.fit_transform(cases_world_death_lstm)
train_data, test_data = cases_world_death_lstm[0:int(len(cases_world_death_lstm)*0.8), :], cases_world_death_lstm[int(len(cases_world_death_lstm)*0.8):len(cases_world_death_lstm), :]
train_x, train_y = prepare_dataset(train_data,10)
test_x, test_y = prepare_dataset(test_data,10)
train_x = np.reshape(train_x, (train_x.shape[0], train_x.shape[1],1))
test_x = np.reshape(test_x, (test_x.shape[0], test_x.shape[1],1))
model = Sequential()
model.add(LSTM(units = 60,activation = 'relu', input_shape = (train_x.shape[1], 1)))
model.add(Dense(units = 1))
model.compile(optimizer = 'adam', loss = 'mean_squared_error')
history = model.fit(train_x, train_y, epochs = 25, batch_size = 15, validation_data = (test_x, test_y), verbose =1)
test_predict = model.predict(test_x)
test_predict = sc.inverse_transform(test_predict)
test_y = sc.inverse_transform([test_y])
print('Test Mean Absolute Error:', mean_absolute_error(test_y[0], test_predict[:,0]))
print('Test Root Mean Squared Error:',np.sqrt(mean_squared_error(test_y[0], test_predict[:,0])))
len(train_data)
# For Predicting Future Prices
Date1 = pd.date_range('2020-07-07', periods=85, freq='D')
columns = ['Date','Price']    
df2 = pd.DataFrame(columns=columns)
df2['Price'] = pd.to_numeric(df2['Price'])
df2["Date"] = pd.to_datetime(Date1)
df2 = df2.fillna(0)
#Remove Weekends as in source data and store data frame as Test1 results to be loaded in Test 1
df1 = df2[df2["Date"].dt.weekday < 5]
df1["Date"] = pd.to_datetime(df1["Date"])
df1['Price'] = pd.to_numeric(df1['Price'])
oil_price_arima=oil_price[(oil_price['Date'] >= '2019-06-17') & (oil_price['Date'] < '2020-07-06')]
oil_price_arima['Date']=pd.to_datetime(oil_price_arima['Date'])
oil_price_arima = oil_price_arima.append(df1,ignore_index=True)
oil_price_arima["Date"] = pd.to_datetime(oil_price_arima["Date"])
oil_price_arima.set_index('Date', inplace=True)
# oil_price_arima.index = pd.to_datetime(oil_price_arima.index)
# oil_price_arima = oil_price_arima.resample('D').ffill().reset_index()
test_data = oil_price_arima.iloc[-84:] 
train_data = oil_price_arima.iloc[:-84] 
autocorrelation_plot(oil_price_arima)
plt.title("Autocorrelation Plot")
plt.show()
plot_pacf(oil_price_arima, lags=31)
plt.title("ACF Plot")
plt.show()
plot_acf(oil_price_arima, lags=31)
plt.title("ACF Plot")
plt.show()
decomposition = sm.tsa.seasonal_decompose(train_data, model='addititve', period=7)
fig = decomposition.plot()
plt.show()
from statsmodels.tsa.arima.model import ARIMA
model = ARIMA(train_data, order=(30,1,5)).fit()
yhat = model.predict(train_data.shape[0], train_data.shape[0]+test_data.shape[0]-1)
final=pd.DataFrame({"Date":test_data.index,"Predicted":yhat, "Actual":test_data["Price"].values})
final.to_csv('Arima_prediction_oil_crude.csv')
final_show=final[(final['Date'] >= '2020-05-04') & (final['Date'] < '2020-06-15')]
final_show.set_index("Date", inplace=True)
final_show["Actual"].plot(label="Actual")
final_show["Predicted"].plot(label="Predicted")
plt.title("Actual vs. Predicted")
plt.legend()
plt.show()
mape_baseline = mean_absolute_error(final_show.Predicted, final_show.Actual)
print('Test MAPE: %.3f' % mape_baseline)
print("Test RMSE",math.sqrt(mean_squared_error(final_show.Actual, final_show.Predicted)))
# For Predicting Future Prices
Date1 = pd.date_range('2020-06-15', periods=60, freq='D')
columns = ['Date','World_total_cases']    
df2 = pd.DataFrame(columns=columns)
df2['World_total_cases'] = pd.to_numeric(df2['World_total_cases'])
df2["Date"] = pd.to_datetime(Date1)
df2 = df2.fillna(0)
#Remove Weekends as in source data and store data frame as Test1 results to be loaded in Test 1
df1 = df2[df2["Date"].dt.weekday < 5]
df1["Date"] = pd.to_datetime(df1["Date"])
df1['World_total_cases'] = pd.to_numeric(df1['World_total_cases'])
cases_world_cases_arima = cases[['Date','World_total_cases']]
cases_world_cases_arima['Date']=pd.to_datetime(cases_world_cases_arima['Date'])
cases_world_cases_arima = cases_world_cases_arima.append(df1,ignore_index=True)
cases_world_cases_arima["Date"] = pd.to_datetime(cases_world_cases_arima["Date"])
cases_world_cases_arima.set_index('Date', inplace=True)
test_data = cases_world_cases_arima.iloc[-72:] 
train_data = cases_world_cases_arima.iloc[:-72] 
autocorrelation_plot(cases_world_cases_arima)
plt.title("Autocorrelation Plot")
plt.show()
plot_pacf(cases_world_cases_arima, lags=31)
plt.title("ACF Plot")
plt.show()
plot_acf(cases_world_cases_arima, lags=31)
plt.title("ACF Plot")
plt.show()
decomposition = sm.tsa.seasonal_decompose(train_data, model='addititve', period=7)
fig = decomposition.plot()
plt.show()
model = ARIMA(train_data, order=(30, 1,5)).fit()
yhat = model.predict(train_data.shape[0], train_data.shape[0]+test_data.shape[0]-1)
final=pd.DataFrame({"Date":test_data.index,"Predicted":yhat, "Actual":test_data["World_total_cases"].values})
final.to_csv('Arima_world_total_cases.csv')
final_show=final[(final['Date'] >= '2020-05-04') & (final['Date'] < '2020-06-15')]
final_show.set_index("Date", inplace=True)
final_show["Actual"].plot(label="Actual")
final_show["Predicted"].plot(label="Predicted")
plt.title("Actual vs. Predicted")
plt.legend()
plt.show()
mape_baseline = mean_absolute_error(final_show.Predicted, final_show.Actual)
print('Test MAPE: %.3f' % mape_baseline)
print("Test RMSE",math.sqrt(mean_squared_error(final_show.Actual, final_show.Predicted)))
# For Predicting Future Prices
Date1 = pd.date_range('2020-06-15', periods=60, freq='D')
columns = ['Date','World_total_deaths']    
df2 = pd.DataFrame(columns=columns)
df2['World_total_deaths'] = pd.to_numeric(df2['World_total_deaths'])
df2["Date"] = pd.to_datetime(Date1)
df2 = df2.fillna(0)
#Remove Weekends as in source data and store data frame as Test1 results to be loaded in Test 1
df1 = df2[df2["Date"].dt.weekday < 5]
df1["Date"] = pd.to_datetime(df1["Date"])
df1['World_total_deaths'] = pd.to_numeric(df1['World_total_deaths'])
cases_world_death_arima = cases[['Date','World_total_deaths']]
cases_world_death_arima['Date']=pd.to_datetime(cases_world_death_arima['Date'])
cases_world_death_arima = cases_world_death_arima.append(df1,ignore_index=True)
cases_world_death_arima["Date"] = pd.to_datetime(cases_world_death_arima["Date"])
cases_world_death_arima.set_index('Date', inplace=True)
test_data = cases_world_death_arima.iloc[-72:] 
train_data = cases_world_death_arima.iloc[:-72] 
autocorrelation_plot(cases_world_death_arima)
plt.title("Autocorrelation Plot")
plt.show()
plot_pacf(cases_world_death_arima, lags=31)
plt.title("ACF Plot")
plt.show()
plot_acf(cases_world_death_arima, lags=31)
plt.title("ACF Plot")
plt.show()
decomposition = sm.tsa.seasonal_decompose(train_data, model='addititve', period=7)
fig = decomposition.plot()
plt.show()
model = ARIMA(train_data, order=(30, 1,5)).fit()
yhat = model.predict(train_data.shape[0], train_data.shape[0]+test_data.shape[0]-1)
final=pd.DataFrame({"Date":test_data.index,"Predicted":yhat, "Actual":test_data["World_total_deaths"].values})
final.to_csv('Arima_world_death.csv')
final_show=final[(final['Date'] >= '2020-05-04') & (final['Date'] < '2020-06-15')]
final_show.set_index("Date", inplace=True)
final_show["Actual"].plot(label="Actual")
final_show["Predicted"].plot(label="Predicted")
plt.title("Actual vs. Predicted")
plt.legend()
plt.show()
mape_baseline = mean_absolute_error(final_show.Predicted, final_show.Actual)
print('Test MAPE: %.3f' % mape_baseline)
print("Test RMSE",math.sqrt(mean_squared_error(final_show.Actual, final_show.Predicted)))
cases_world_total_new_lstm = cases[['Date','World_total_cases','World_total_deaths','Price']]
cases_world_total_new_lstm=cases_world_total_new_lstm[(cases_world_total_new_lstm['Date'] >= '2019-12-31') & (cases_world_total_new_lstm['Date'] < '2020-07-06')]
cases_world_total_new_lstm['Date'] = pd.to_datetime(cases_world_total_new_lstm['Date'])
new_df = pd.DataFrame()
new_df["Dates"] = pd.date_range(start='2020-01-01', end='2020-07-06')
new_df["Dates"] = pd.to_datetime(new_df["Dates"])
# cases_world_total_new_lstm_new = pd.concat([new_df,cases_world_total_new_lstm], join="outer", axis=1)
cases_world_total_new_lstm_new = new_df.merge(cases_world_total_new_lstm, left_on="Dates",right_on='Date', how = 'left')
cases_world_total_new_lstm_new.drop(columns='Date',inplace=True)
cases_world_total_new_lstm_new = cases_world_total_new_lstm_new.fillna(0)
cases_world_total_new_lstm_new.tail(21)
sat_day=3
for i in range(27):
    sun_day= sat_day+1
    prev_day =sat_day-1
    next_day =sat_day+2
#     print(cases_world_total_new_lstm_new.iloc[next_day,cases_world_total_new_lstm_new.columns.get_loc("China_total_cases")])
#     print(cases_world_total_new_lstm_new.iloc[prev_day,cases_world_total_new_lstm_new.columns.get_loc("China_total_cases")])
    if (cases_world_total_new_lstm_new.iloc[next_day,cases_world_total_new_lstm_new.columns.get_loc("World_total_cases")] != 0 or cases_world_total_new_lstm_new.iloc[prev_day,cases_world_total_new_lstm_new.columns.get_loc("World_total_cases")] != 0):
        Threshold = (cases_world_total_new_lstm_new.iloc[next_day,cases_world_total_new_lstm_new.columns.get_loc("World_total_cases")] + cases_world_total_new_lstm_new.iloc[prev_day,cases_world_total_new_lstm_new.columns.get_loc("World_total_cases")]) / 2
        Threshold_sat = (Threshold + cases_world_total_new_lstm_new.iloc[prev_day,cases_world_total_new_lstm_new.columns.get_loc("World_total_cases")]) / 2
        Threshold_sun = (Threshold + cases_world_total_new_lstm_new.iloc[next_day,cases_world_total_new_lstm_new.columns.get_loc("World_total_cases")]) / 2
        cases_world_total_new_lstm_new.iloc[sat_day,cases_world_total_new_lstm_new.columns.get_loc("World_total_cases")] = Threshold_sat
        cases_world_total_new_lstm_new.iloc[sun_day,cases_world_total_new_lstm_new.columns.get_loc("World_total_cases")] = Threshold_sun
    else:
        cases_world_total_new_lstm_new.iloc[sat_day,cases_world_total_new_lstm_new.columns.get_loc("World_total_cases")] = cases_world_total_new_lstm_new.iloc[prev_day,cases_world_total_new_lstm_new.columns.get_loc("World_total_cases")]
        cases_world_total_new_lstm_new.iloc[sun_day,cases_world_total_new_lstm_new.columns.get_loc("World_total_cases")] = cases_world_total_new_lstm_new.iloc[prev_day,cases_world_total_new_lstm_new.columns.get_loc("World_total_cases")]
    sat_day=sat_day+7

# sat_day=3
# for i in range(23):
#     sun_day= sat_day+1
#     prev_day =sat_day-1
#     next_day =sat_day+2
# #     print(cases_world_total_new_lstm_new.iloc[next_day,cases_world_total_new_lstm_new.columns.get_loc("China_total_cases")])
# #     print(cases_world_total_new_lstm_new.iloc[prev_day,cases_world_total_new_lstm_new.columns.get_loc("China_total_cases")])
#     if (cases_world_total_new_lstm_new.iloc[next_day,cases_world_total_new_lstm_new.columns.get_loc("Italy_total_cases")] != 0 or cases_world_total_new_lstm_new.iloc[prev_day,cases_world_total_new_lstm_new.columns.get_loc("Italy_total_cases")] != 0):
#         Threshold = (cases_world_total_new_lstm_new.iloc[next_day,cases_world_total_new_lstm_new.columns.get_loc("Italy_total_cases")] + cases_world_total_new_lstm_new.iloc[prev_day,cases_world_total_new_lstm_new.columns.get_loc("Italy_total_cases")]) / 2
#         Threshold_sat = (Threshold + cases_world_total_new_lstm_new.iloc[prev_day,cases_world_total_new_lstm_new.columns.get_loc("Italy_total_cases")]) / 2
#         Threshold_sun = (Threshold + cases_world_total_new_lstm_new.iloc[next_day,cases_world_total_new_lstm_new.columns.get_loc("Italy_total_cases")]) / 2
#         cases_world_total_new_lstm_new.iloc[sat_day,cases_world_total_new_lstm_new.columns.get_loc("Italy_total_cases")] = Threshold_sat
#         cases_world_total_new_lstm_new.iloc[sun_day,cases_world_total_new_lstm_new.columns.get_loc("Italy_total_cases")] = Threshold_sun
#     else:
#         cases_world_total_new_lstm_new.iloc[sat_day,cases_world_total_new_lstm_new.columns.get_loc("Italy_total_cases")] = cases_world_total_new_lstm_new.iloc[prev_day,cases_world_total_new_lstm_new.columns.get_loc("Italy_total_cases")]
#         cases_world_total_new_lstm_new.iloc[sun_day,cases_world_total_new_lstm_new.columns.get_loc("Italy_total_cases")] = cases_world_total_new_lstm_new.iloc[prev_day,cases_world_total_new_lstm_new.columns.get_loc("Italy_total_cases")]
#     sat_day=sat_day+7
# sat_day=3
# for i in range(23):
#     sun_day= sat_day+1
#     prev_day =sat_day-1
#     next_day =sat_day+2
# #     print(cases_world_total_new_lstm_new.iloc[next_day,cases_world_total_new_lstm_new.columns.get_loc("China_total_cases")])
# #     print(cases_world_total_new_lstm_new.iloc[prev_day,cases_world_total_new_lstm_new.columns.get_loc("China_total_cases")])
#     if (cases_world_total_new_lstm_new.iloc[next_day,cases_world_total_new_lstm_new.columns.get_loc("France_total_cases")] != 0 or cases_world_total_new_lstm_new.iloc[prev_day,cases_world_total_new_lstm_new.columns.get_loc("France_total_cases")] != 0):
#         Threshold = (cases_world_total_new_lstm_new.iloc[next_day,cases_world_total_new_lstm_new.columns.get_loc("France_total_cases")] + cases_world_total_new_lstm_new.iloc[prev_day,cases_world_total_new_lstm_new.columns.get_loc("France_total_cases")]) / 2
#         Threshold_sat = (Threshold + cases_world_total_new_lstm_new.iloc[prev_day,cases_world_total_new_lstm_new.columns.get_loc("France_total_cases")]) / 2
#         Threshold_sun = (Threshold + cases_world_total_new_lstm_new.iloc[next_day,cases_world_total_new_lstm_new.columns.get_loc("France_total_cases")]) / 2
#         cases_world_total_new_lstm_new.iloc[sat_day,cases_world_total_new_lstm_new.columns.get_loc("France_total_cases")] = Threshold_sat
#         cases_world_total_new_lstm_new.iloc[sun_day,cases_world_total_new_lstm_new.columns.get_loc("France_total_cases")] = Threshold_sun
#     else:
#         cases_world_total_new_lstm_new.iloc[sat_day,cases_world_total_new_lstm_new.columns.get_loc("France_total_cases")] = cases_world_total_new_lstm_new.iloc[prev_day,cases_world_total_new_lstm_new.columns.get_loc("France_total_cases")]
#         cases_world_total_new_lstm_new.iloc[sun_day,cases_world_total_new_lstm_new.columns.get_loc("France_total_cases")] = cases_world_total_new_lstm_new.iloc[prev_day,cases_world_total_new_lstm_new.columns.get_loc("France_total_cases")]
#     sat_day=sat_day+7
# sat_day=3
# for i in range(23):
#     sun_day= sat_day+1
#     prev_day =sat_day-1
#     next_day =sat_day+2
# #     print(cases_world_total_new_lstm_new.iloc[next_day,cases_world_total_new_lstm_new.columns.get_loc("China_total_cases")])
# #     print(cases_world_total_new_lstm_new.iloc[prev_day,cases_world_total_new_lstm_new.columns.get_loc("China_total_cases")])
#     if (cases_world_total_new_lstm_new.iloc[next_day,cases_world_total_new_lstm_new.columns.get_loc("UnitedStates_total_cases")] != 0 or cases_world_total_new_lstm_new.iloc[prev_day,cases_world_total_new_lstm_new.columns.get_loc("UnitedStates_total_cases")] != 0):
#         Threshold = (cases_world_total_new_lstm_new.iloc[next_day,cases_world_total_new_lstm_new.columns.get_loc("UnitedStates_total_cases")] + cases_world_total_new_lstm_new.iloc[prev_day,cases_world_total_new_lstm_new.columns.get_loc("UnitedStates_total_cases")]) / 2
#         Threshold_sat = (Threshold + cases_world_total_new_lstm_new.iloc[prev_day,cases_world_total_new_lstm_new.columns.get_loc("UnitedStates_total_cases")]) / 2
#         Threshold_sun = (Threshold + cases_world_total_new_lstm_new.iloc[next_day,cases_world_total_new_lstm_new.columns.get_loc("UnitedStates_total_cases")]) / 2
#         cases_world_total_new_lstm_new.iloc[sat_day,cases_world_total_new_lstm_new.columns.get_loc("UnitedStates_total_cases")] = Threshold_sat
#         cases_world_total_new_lstm_new.iloc[sun_day,cases_world_total_new_lstm_new.columns.get_loc("UnitedStates_total_cases")] = Threshold_sun
#     else:
#         cases_world_total_new_lstm_new.iloc[sat_day,cases_world_total_new_lstm_new.columns.get_loc("UnitedStates_total_cases")] = cases_world_total_new_lstm_new.iloc[prev_day,cases_world_total_new_lstm_new.columns.get_loc("UnitedStates_total_cases")]
#         cases_world_total_new_lstm_new.iloc[sun_day,cases_world_total_new_lstm_new.columns.get_loc("UnitedStates_total_cases")] = cases_world_total_new_lstm_new.iloc[prev_day,cases_world_total_new_lstm_new.columns.get_loc("UnitedStates_total_cases")]
#     sat_day=sat_day+7
len(cases_world_total_new_lstm_new)
sat_day=3
for i in range(27):
    sun_day= sat_day+1
    prev_day =sat_day-1
    next_day =sat_day+2
    Threshold = (cases_world_total_new_lstm_new.iloc[next_day,cases_world_total_new_lstm_new.columns.get_loc("Price")] + cases_world_total_new_lstm_new.iloc[prev_day,cases_world_total_new_lstm_new.columns.get_loc("Price")]) / 2
    Threshold_sat = (Threshold + cases_world_total_new_lstm_new.iloc[prev_day,cases_world_total_new_lstm_new.columns.get_loc("Price")]) / 2
    Threshold_sun = (Threshold + cases_world_total_new_lstm_new.iloc[next_day,cases_world_total_new_lstm_new.columns.get_loc("Price")]) / 2
    cases_world_total_new_lstm_new.iloc[sat_day,cases_world_total_new_lstm_new.columns.get_loc("Price")] = Threshold_sat
    cases_world_total_new_lstm_new.iloc[sun_day,cases_world_total_new_lstm_new.columns.get_loc("Price")] = Threshold_sun
    sat_day=sat_day+7
len(cases_world_total_new_lstm_new)
sat_day=3
for i in range(27):
    sun_day= sat_day+1
    prev_day =sat_day-1
    next_day =sat_day+2
    
    Threshold = (cases_world_total_new_lstm_new.iloc[next_day,cases_world_total_new_lstm_new.columns.get_loc("World_total_deaths")] + cases_world_total_new_lstm_new.iloc[prev_day,cases_world_total_new_lstm_new.columns.get_loc("World_total_deaths")]) / 2
    Threshold_sat = (Threshold + cases_world_total_new_lstm_new.iloc[prev_day,cases_world_total_new_lstm_new.columns.get_loc("World_total_deaths")]) / 2
    Threshold_sun = (Threshold + cases_world_total_new_lstm_new.iloc[next_day,cases_world_total_new_lstm_new.columns.get_loc("World_total_deaths")]) / 2
    cases_world_total_new_lstm_new.iloc[sat_day,cases_world_total_new_lstm_new.columns.get_loc("World_total_deaths")] = Threshold_sat
    cases_world_total_new_lstm_new.iloc[sun_day,cases_world_total_new_lstm_new.columns.get_loc("World_total_deaths")] = Threshold_sun
    sat_day=sat_day+7
# sat_day=3
# for i in range(23):
#     sun_day= sat_day+1
#     prev_day =sat_day-1
#     next_day =sat_day+2
    
#     Threshold = (cases_world_total_new_lstm_new.iloc[next_day,cases_world_total_new_lstm_new.columns.get_loc("Italy_total_deaths")] + cases_world_total_new_lstm_new.iloc[prev_day,cases_world_total_new_lstm_new.columns.get_loc("Italy_total_deaths")]) / 2
#     Threshold_sat = (Threshold + cases_world_total_new_lstm_new.iloc[prev_day,cases_world_total_new_lstm_new.columns.get_loc("Italy_total_deaths")]) / 2
#     Threshold_sun = (Threshold + cases_world_total_new_lstm_new.iloc[next_day,cases_world_total_new_lstm_new.columns.get_loc("Italy_total_deaths")]) / 2
#     cases_world_total_new_lstm_new.iloc[sat_day,cases_world_total_new_lstm_new.columns.get_loc("Italy_total_deaths")] = Threshold_sat
#     cases_world_total_new_lstm_new.iloc[sun_day,cases_world_total_new_lstm_new.columns.get_loc("Italy_total_deaths")] = Threshold_sun
#     sat_day=sat_day+7
# sat_day=3
# for i in range(23):
#     sun_day= sat_day+1
#     prev_day =sat_day-1
#     next_day =sat_day+2
    
#     Threshold = (cases_world_total_new_lstm_new.iloc[next_day,cases_world_total_new_lstm_new.columns.get_loc("France_total_deaths")] + cases_world_total_new_lstm_new.iloc[prev_day,cases_world_total_new_lstm_new.columns.get_loc("France_total_deaths")]) / 2
#     Threshold_sat = (Threshold + cases_world_total_new_lstm_new.iloc[prev_day,cases_world_total_new_lstm_new.columns.get_loc("France_total_deaths")]) / 2
#     Threshold_sun = (Threshold + cases_world_total_new_lstm_new.iloc[next_day,cases_world_total_new_lstm_new.columns.get_loc("France_total_deaths")]) / 2
#     cases_world_total_new_lstm_new.iloc[sat_day,cases_world_total_new_lstm_new.columns.get_loc("France_total_deaths")] = Threshold_sat
#     cases_world_total_new_lstm_new.iloc[sun_day,cases_world_total_new_lstm_new.columns.get_loc("France_total_deaths")] = Threshold_sun
#     sat_day=sat_day+7
# sat_day=3
# for i in range(23):
#     sun_day= sat_day+1
#     prev_day =sat_day-1
#     next_day =sat_day+2
    
#     Threshold = (cases_world_total_new_lstm_new.iloc[next_day,cases_world_total_new_lstm_new.columns.get_loc("UnitedStates_total_deaths")] + cases_world_total_new_lstm_new.iloc[prev_day,cases_world_total_new_lstm_new.columns.get_loc("UnitedStates_total_deaths")]) / 2
#     Threshold_sat = (Threshold + cases_world_total_new_lstm_new.iloc[prev_day,cases_world_total_new_lstm_new.columns.get_loc("UnitedStates_total_deaths")]) / 2
#     Threshold_sun = (Threshold + cases_world_total_new_lstm_new.iloc[next_day,cases_world_total_new_lstm_new.columns.get_loc("UnitedStates_total_deaths")]) / 2
#     cases_world_total_new_lstm_new.iloc[sat_day,cases_world_total_new_lstm_new.columns.get_loc("UnitedStates_total_deaths")] = Threshold_sat
#     cases_world_total_new_lstm_new.iloc[sun_day,cases_world_total_new_lstm_new.columns.get_loc("UnitedStates_total_deaths")] = Threshold_sun
#     sat_day=sat_day+7
len(cases_world_total_new_lstm_new)
# cases_world_total_new_lstm.reset_index(drop=True,inplace=True)
# cases_world_total_new_lstm_new.to_csv('cases.csv')
# cases_world_total_new_lstm_new['prev_price'] = cases_world_total_new_lstm_new['Price'].shift(1)
# # cases_world_total_new_lstm_new = cases_world_total_new_lstm_new.dropna()
# cases_world_total_new_lstm_new['diff'] = (cases_world_total_new_lstm_new['Price'] - cases_world_total_new_lstm_new['prev_price'])
# df_supervised = cases_world_total_new_lstm_new.drop(['prev_price'],axis=1)
# for inc in range(1,13):
#     field_name = 'lag_' + str(inc)
#     df_supervised[field_name] = df_supervised['diff'].shift(inc)
#     #drop null values
# # df_supervised = df_supervised.dropna().reset_index(drop=True)
# df_supervised['prev_China_total_cases'] = df_supervised['China_total_cases'].shift(1)
# # df_supervised = df_supervised.dropna()
# df_supervised['diff_China_total_cases'] = (df_supervised['China_total_cases'] - df_supervised['prev_China_total_cases'])
# df_supervised_1 = df_supervised.drop(['prev_China_total_cases'],axis=1)
# for inc in range(1,13):
#     field_name = 'China_total_cases_lag_' + str(inc)
#     df_supervised_1[field_name] = df_supervised_1['diff_China_total_cases'].shift(inc)
#     #drop null values
# # df_supervised_1 = df_supervised_1.dropna().reset_index(drop=True)
# df_supervised_1['prev_China_total_deaths'] = df_supervised_1['China_total_deaths'].shift(1)
# # df_supervised_1 = df_supervised_1.dropna()
# df_supervised_1['diff_China_total_deaths'] = (df_supervised_1['China_total_deaths'] - df_supervised_1['prev_China_total_deaths'])
# df_supervised_2 = df_supervised_1.drop(['prev_China_total_deaths'],axis=1)
# for inc in range(1,13):
#     field_name = 'China_total_deaths_lag_' + str(inc)
#     df_supervised_2[field_name] = df_supervised_2['diff_China_total_deaths'].shift(inc)
#     #drop null values
# # df_supervised_2 = df_supervised_2.dropna().reset_index(drop=True)
# len(df_supervised_2)
china_feature_matrix = pd.read_csv(r'C:\Users\205694\Desktop\Sentiment Analysis\Crude Oil\ntt-data-global-ai-challenge-06-2020\Feature_Matrix_ALL_resize.csv')
china_feature_matrix['Dates'] = pd.date_range('2020-01-01', periods=188, freq='D')
china_feature_matrix.drop(columns=['PC8','PC9','PC10'],inplace=True)
# china_feature_matrix['Dates'] = pd.date_range(start='2020-01-01', end='2020-06-08')
china_feature_matrix
# final_china = pd.merge(df_supervised_2, china_feature_matrix, on="Dates")
final_china_1 = pd.merge(cases_world_total_new_lstm_new, china_feature_matrix, on='Dates')
# final_china = final_china.fillna(0)
final_china_1 = final_china_1.fillna(0)
final_china_1.head(5)
final_china_1 = final_china_1[['Dates', 'World_total_cases', 'World_total_deaths', 'PC1', 'PC2', 'PC3','PC4','PC5','PC6','PC7','Price']]
# For Predicting Future Prices
Date1 = pd.date_range('2020-07-07', periods=60, freq='D')
columns = ['Dates', 'World_total_cases', 'World_total_deaths', 'Price', 'PC1',
       'PC2', 'PC3','PC4','PC5','PC6','PC7']
# , 'PC4', 'PC5', 'PC6', 'PC7', 'PC8', 'PC9', 'PC10']    
df2 = pd.DataFrame(columns=columns)
df2['World_total_cases'] = pd.to_numeric(df2['World_total_cases'])
df2["Dates"] = pd.to_datetime(Date1)
df2['World_total_deaths'] = pd.to_numeric(df2['World_total_deaths'])
df2["Price"] = pd.to_numeric(df2['Price'])
df2['PC1'] = pd.to_numeric(df2['PC1'])
df2["PC2"] = pd.to_numeric(df2['PC2'])
df2['PC3'] = pd.to_numeric(df2['PC3'])
df2["PC4"] = pd.to_numeric(df2['PC4'])
df2['PC5'] = pd.to_numeric(df2['PC5'])
df2["PC6"] = pd.to_numeric(df2['PC6'])
df2['PC7'] = pd.to_numeric(df2['PC7'])
# df2["PC8"] = pd.to_numeric(df2['PC8'])
# df2['PC9'] = pd.to_numeric(df2['PC9'])
# df2["PC10"] = pd.to_numeric(df2['PC10'])
df2 = df2.fillna(0)
#Remove Weekends as in source data and store data frame as Test1 results to be loaded in Test 1
# df1 = df2[df2["Date"].dt.weekday < 5]
# df1["Date"] = pd.to_datetime(df1["Date"])
# df1['World_total_deaths'] = pd.to_numeric(df1['World_total_deaths'])
final_china_1 = final_china_1.append(df2,ignore_index=True)
final_china_1["Dates"] = pd.to_datetime(final_china_1["Dates"])
len(final_china_1)
final_china_1 = final_china_1.fillna(0)
final_china_1.set_index('Dates', inplace=True)
final_china_1
test_ind = 85
train = final_china_1.iloc[:-test_ind]
test = final_china_1.iloc[-test_ind:]
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(train)
scaled_train = scaler.transform(train)
scaled_test = scaler.transform(test)
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
length = 5 # Length of the output sequences (in number of timesteps)
batch_size = 1 #Number of timeseries samples in each batch
generator = TimeseriesGenerator(scaled_train, scaled_train, length=length, batch_size=batch_size)
X,y = generator[0]
scaled_train.shape
model = Sequential()

# Simple RNN layer
model.add(LSTM(100,input_shape=(length,scaled_train.shape[1])))

# Final Prediction (one neuron per feature)
model.add(Dense(scaled_train.shape[1]))

model.compile(optimizer='adam', loss='mse')
model.summary()
from tensorflow.keras.callbacks import EarlyStopping
early_stop = EarlyStopping(monitor='val_loss',patience=1)
validation_generator = TimeseriesGenerator(scaled_test,scaled_test,length=length, batch_size=batch_size)
model.fit_generator(generator,epochs=10,
                    validation_data=validation_generator,
                   callbacks=[early_stop])
losses = pd.DataFrame(model.history.history)
first_eval_batch = scaled_train[-length:]
first_eval_batch = first_eval_batch.reshape((1, length, scaled_train.shape[1]))
model.predict(first_eval_batch)
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
rmse = np.sqrt(mean_squared_error(true_predictions['Price'][:21], test['Price'][:21]))
print('Test RMSE: %.3f' % rmse)
y=[x for x in range(21)]
plt.figure(figsize=(8,4))
plt.plot(y, test['Price'][:21], marker='.', label="actual")
plt.plot(y, true_predictions['Price'][:21], 'r', label="prediction")
plt.tight_layout()
plt.subplots_adjust(left=0.07)
plt.ylabel('Price', size=15)
plt.xlabel('Time step', size=15)
plt.legend(fontsize=15)
plt.show();
Date1 = pd.date_range('2020-06-12', periods=85, freq='D')
columns = ['Dates','Actual Price','Predicted Price']    
df_final = pd.DataFrame(columns=columns)
df_final['Actual Price'] = test['Price']
df_final['Predicted Price'] = true_predictions['Price'].values
df_final["Dates"] = pd.to_datetime(Date1)
df_final
df_final.to_csv('multi_lstm.csv')