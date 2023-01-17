#!pip uninstall statsmodels
 #!pip install statsmodels --upgraded
#!pip install pyramid-arima
!pip install pmdarima # libreria con la función auto_arima()
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import statsmodels as sm

from statsmodels.tsa.stattools import adfuller

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

from statsmodels.tsa.seasonal import seasonal_decompose

from statsmodels.tsa.arima_model import ARIMA

from sklearn.metrics import mean_squared_error

from scipy import stats

from matplotlib import pyplot

from pmdarima import auto_arima

from sklearn.preprocessing import MinMaxScaler

from keras.preprocessing.sequence import TimeseriesGenerator

from keras.models import Sequential

from keras.layers import Dense

from keras.layers import LSTM

from fbprophet import Prophet





%matplotlib inline

sns.set()
url = 'https://raw.githubusercontent.com/sergiomora03/AdvancedMethodsDataAnalysis/master/datasets_56102_107707_monthly-beer-production-in-austr.csv'

data = pd.read_csv(url)

data.head()
#Descripción de la base de datos

data.info()
data.Month = pd.to_datetime(data.Month)

data.set_index('Month', inplace=True)

data.head()
#Producción en logaritmo

data['Log_Production']=np.log(data['Monthly beer production'])
#Evolución de producción de cerveza en Australia durante el periodo de análisis (niveles y logaritmo)

data[['Monthly beer production']].plot(figsize=(20,5), linewidth=2, fontsize=10)

plt.xlabel('time', fontsize=15);

data[['Log_Production']].plot(figsize=(20,5), linewidth=2, fontsize=10)

plt.xlabel('time', fontsize=15);
res = seasonal_decompose(data['Monthly beer production'], model='additive',freq=12)



def plotseasonal(res, axes ):

    res.observed.plot(ax=axes[0], legend=False)

    axes[0].set_ylabel('Observed')

    res.trend.plot(ax=axes[1], legend=False)

    axes[1].set_ylabel('Trend')

    res.seasonal.plot(ax=axes[2], legend=False)

    axes[2].set_ylabel('Seasonal')

    res.resid.plot(ax=axes[3], legend=False)

    axes[3].set_ylabel('Residual')



fig, axes = plt.subplots(ncols=1, nrows=4, sharex=True, figsize=(20,5))



plotseasonal(res, axes)



plt.tight_layout()

plt.show()

res = seasonal_decompose(data['Log_Production'], model='additive',freq=12)



fig, axes = plt.subplots(ncols=1, nrows=4, sharex=True, figsize=(20,5))



plotseasonal(res, axes)



plt.tight_layout()

plt.show()
beer=data['Monthly beer production']

Log_beer=data['Log_Production']
#Primera diferencia de la serie (niveles)

beer.diff().plot(figsize=(20,5), linewidth=2, fontsize=10)

plt.xlabel('time', fontsize=10);
#Primera diferencia de la serie (logaritmo)

Log_beer.diff().plot(figsize=(20,5), linewidth=2, fontsize=10)
#Prueba Dickey- Fuller

result = adfuller(data['Log_Production'])

print('ADF Statistic: %f' % result[0])

print('p-value: %f' % result[1])

print('Critical Values:')

for key, value in result[4].items():

    print('\t%s: %.3f' % (key, value))
#Prueba Dickey- Fuller - primera diferencia

result = adfuller(data['Log_Production'].diff().iloc[1:])

print('ADF Statistic: %f' % result[0])

print('p-value: %f' % result[1])

print('Critical Values:')

for key, value in result[4].items():

    print('\t%s: %.3f' % (key, value))
#Autocorrelación de la serie en análisis

plt.figure(figsize=(20,5))

pd.plotting.autocorrelation_plot(data['Log_Production']);
#Autocorrelación de la primera diferencia de la serie en análisis

plt.figure(figsize=(20,5))

pd.plotting.autocorrelation_plot(data['Log_Production'].diff().iloc[1:]);
#Autocorrelación simple y parcial serie producción cerveza

fig, ax = plt.subplots(1,2,figsize=(20,5))

plot_acf(data['Log_Production'], ax=ax[0])

plot_pacf(data['Log_Production'], ax=ax[1])

plt.show()
#Autocorrelación simple y parcial primera diferencia de serie producción cerveza

fig, ax = plt.subplots(1,2,figsize=(20,5))

plot_acf(data['Log_Production'].diff().iloc[1:], ax=ax[0])

plot_pacf(data['Log_Production'].diff().iloc[1:], ax=ax[1])

plt.show()
X = data['Log_Production'].values

size = int(len(X) * 0.9)

train, test = X[0:size], X[size:len(X)]





def ARIMA_FUNCTION(p,q):

	for t in range(len(test)):

		model_1 = ARIMA(history, order=(p,1,q))

		model_fit_1 = model_1.fit(disp=0)

		output = model_fit_1.forecast()

		yhat = output[0]

		predictions.append(yhat)

		obs = test[t]

		history.append(obs)

	return mean_squared_error(test, predictions)**0.5
results=[]

for i in range(3):

    for j in range(4):

        if (i>j or i==0) and (i-j<3) :

            predictions = list()

            history = [x for x in train]

            results.append([i,j,ARIMA_FUNCTION(i,j)])

            print((i,j))
results
model = ARIMA(train, order=(0,1,2))

model_fit = model.fit(disp=0)

print(model_fit.summary())
residuals = pd.DataFrame(model_fit.resid)

residuals.plot(figsize=(20,5))

plt.show()
residuals.plot(kind='kde', figsize=(20,5))

plt.show()

print(residuals.describe())
print("KS P-value = "+str(round(stats.kstest(residuals, 'norm')[1], 10)))
history = [x for x in train]

predictions = list()



for t in range(len(test)):

    model_1 = ARIMA(history, order=(0,1,2))

    model_fit_1 = model_1.fit(disp=0)

    output = model_fit_1.forecast()

    yhat = output[0]

    predictions.append(yhat)

    obs = test[t]

    history.append(obs)

    print('predicted=%f, expected=%f' % (yhat, obs))
error_ARIMA = mean_squared_error(test, predictions)**0.5

print('Test RMSE: %.3f' % error_ARIMA)
RollBack=pd.concat([pd.DataFrame({'TEST':test}),pd.DataFrame({'ARIMA':np.concatenate(predictions, axis=0)})],axis=1)

RollBack.head()
RollBack.plot(figsize=(20,5), linewidth=2, fontsize=10)

plt.xlabel('time', fontsize=15);
stepwise_model = auto_arima(train, 

                            start_p=0,

                            start_q=0, 

                            max_p=5, 

                            max_d=2, 

                            max_q=5, 

                            start_P=1,

                            start_Q=1, 

                            max_P=2, 

                            max_D=2, 

                            max_Q=2, 

                            max_order=10,

                            m=12,

                            seasonal=True,

                            trace=True,

                            error_action='ignore',  

                            suppress_warnings=True, 

                            stepwise=True)

print(stepwise_model.aic())
mod = sm.tsa.statespace.sarimax.SARIMAX(train, trend='n', order=(1,1,1), seasonal_order=(2,0,2,12))

results = mod.fit()

results.summary()
#results.resid

residuals1 = pd.DataFrame(results.resid[1:])

residuals1.plot(figsize=(20,5))

plt.show()
residuals1.plot(kind='kde', figsize=(20,5))

plt.show()

print(residuals1.describe())
print("KS P-value = "+str(round(stats.kstest(residuals1, 'norm')[1], 10)))

print("D’Agostino and Pearson’s P-value = "+str(round(stats.normaltest(residuals1, axis=0)[1][0], 6)))
X = data['Log_Production'].values

size = int(len(X) * 0.9)

train, test = X[0:size], X[size:len(X)]

history = [x for x in train]

predictions = list()

for t in range(len(test)):

	model = sm.tsa.statespace.sarimax.SARIMAX(history, trend='n', order=(1,1,1), seasonal_order=(2,0,2,12))

	model_fit = model.fit(disp=0)

	output = model_fit.forecast()

	yhat = output[0]

	predictions.append(yhat)

	obs = test[t]

	history.append(obs)

error_SARIMA = mean_squared_error(test, predictions)**0.5

print('Test RMSE: %.3f' % error_SARIMA)
RollBack=pd.concat([RollBack,pd.DataFrame({'SARIMA':predictions})],axis=1)

RollBack.head()
RollBack.plot(figsize=(20,5), linewidth=2, fontsize=10)

plt.xlabel('time', fontsize=15);
data_pf = pd.DataFrame({'ds': data.Log_Production.index[:], 'y': data.Log_Production})

data_pf.head()
X = data_pf.y

Y = data_pf.ds

size = int(len(X) * 0.9)

train_X, test_X = X[0:size], X[size:len(X)]

train_Y, test_Y = Y[0:size], Y[size:len(Y)]

    

Train = pd.concat([train_Y,train_X], axis=1)

Test = pd.concat([test_Y,test_X], axis=1)
predictions = list()

    

def rolling_forecast():   

    history = Train.copy()

    

    for t in range(len(test_X)):

        m = Prophet()

        m.fit(history);

        future = m.make_future_dataframe(periods=1, freq='MS')

        forecast = m.predict(future)

        output=forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]

        yhat = output[['yhat']][len(history):].values[0][0]

        predictions.append(yhat)

        obs = pd.DataFrame(Test[['ds','y']].iloc[t])

        history = pd.concat([history, obs.transpose()],axis=0)

        print('predicted=%f, expected=%f' % (yhat, obs.transpose()['y']))



    

    error_PROPHET = mean_squared_error(test_X, predictions)**0.5

    print('Test RMSE: %.3f' % error_PROPHET)
rolling_forecast()
error_PROPHET = mean_squared_error(test_X, predictions) **0.5

print('Test RMSE: %.3f' % error_PROPHET)
RollBack=pd.concat([RollBack,pd.DataFrame({'Prophet':predictions})],axis=1)

RollBack.head()
RollBack[['TEST', 'Prophet']].plot(figsize=(20,5), linewidth=2, fontsize=10)

plt.xlabel('time', fontsize=15);
RollBack.plot(figsize=(20,5), linewidth=2, fontsize=10)

plt.xlabel('time', fontsize=15);
data_LSTM = pd.DataFrame({'Log_Production': data.Log_Production})

data_LSTM.head()
Y = data_LSTM

size = int(len(Y) * 0.9)



train_Y, test_Y = Y[0:size], Y[size:len(Y)]
scaler = MinMaxScaler()

scaler.fit(train_Y)

scaled_train_data = scaler.transform(train_Y)

scaled_test_data = scaler.transform(test_Y)
n_input = 12

n_features= 1

generator = TimeseriesGenerator(scaled_train_data, scaled_train_data, length=n_input, batch_size=1)
lstm_model = Sequential()

lstm_model.add(LSTM(200, input_shape=(n_input, n_features)))

#lstm_model.add(LSTM(units=50, return_sequences = True))

lstm_model.add(Dense(1))

lstm_model.compile(optimizer='adam', loss='mean_squared_error')



lstm_model.summary()
lstm_model.fit_generator(generator,epochs=20)
losses_lstm = lstm_model.history.history['loss']

plt.figure(figsize=(20,5))

plt.xticks(np.arange(0,21,1))

plt.plot(range(len(losses_lstm)),losses_lstm);
lstm_predictions_scaled = list()



batch = scaled_train_data[-n_input:]

current_batch = batch.reshape((1, n_input, n_features))



for i in range(len(test)):   

    lstm_pred = lstm_model.predict(current_batch)[0]

    lstm_predictions_scaled.append(lstm_pred) 

    current_batch = np.append(current_batch[:,1:,:],[[lstm_pred]],axis=1)
lstm_predictions = scaler.inverse_transform(lstm_predictions_scaled)
error_LSTM = mean_squared_error(test, lstm_predictions) ** 0.5

print('Test RMSE: %.3f' % error_LSTM)
RollBack = pd.concat([RollBack,pd.DataFrame({'LSTM':np.concatenate(lstm_predictions, axis=0)})],axis=1)

RollBack.head()
RollBack[['TEST', 'LSTM']].plot(figsize=(20,5), linewidth=2, fontsize=10)

plt.xlabel('time', fontsize=15);
RollBack.plot(figsize=(20,5), linewidth=2, fontsize=10)

plt.xlabel('time', fontsize=15);
RollBack = pd.concat([RollBack,pd.DataFrame({'Time':data.Log_Production.index[size:]})],axis=1)

RollBack.head()
RollBack.set_index('Time', inplace=True)

RollBack.head()
RollBack.plot(figsize=(20,5), linewidth=2, fontsize=10)

plt.xlabel('time', fontsize=15);
Error = pd.DataFrame({"Models":["ARIMA", "SARIMA", "Prophet", "LSTM"],

                      "RMSE Log" : [error_ARIMA, error_SARIMA, error_PROPHET, error_LSTM]})

Error
print('Test RMSE ARIMA: %.3f' % mean_squared_error(np.exp(RollBack.TEST), np.exp(RollBack.ARIMA)))

print('Test RMSE SARIMA: %.3f' % mean_squared_error(np.exp(RollBack.TEST), np.exp(RollBack.SARIMA)))

print('Test RMSE Prophet: %.3f' % mean_squared_error(np.exp(RollBack.TEST), np.exp(RollBack.Prophet)))

print('Test RMSE LSTM: %.3f' % mean_squared_error(np.exp(RollBack.TEST), np.exp(RollBack.LSTM)))
fig, ax = plt.subplots(1,2,figsize=(20,5))

np.exp(RollBack[['SARIMA','TEST']]).plot(figsize=(20,5), linewidth=2, fontsize=10, ax = ax[0])

np.exp(RollBack[['Prophet','TEST']]).plot(figsize=(20,5), linewidth=2, fontsize=10, ax = ax[1])

plt.xlabel('time', fontsize=15);

#plt.show()