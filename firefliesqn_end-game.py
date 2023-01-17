import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight') 

%matplotlib inline

from pylab import rcParams

from plotly import tools

import plotly.plotly as py

from plotly.offline import init_notebook_mode, iplot

init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.figure_factory as ff

import statsmodels.api as sm

from numpy.random import normal, seed

from scipy.stats import norm

from statsmodels.tsa.arima_model import ARMA

from statsmodels.tsa.stattools import adfuller

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

from statsmodels.tsa.arima_process import ArmaProcess

from statsmodels.tsa.arima_model import ARIMA

import math

from sklearn.metrics import mean_squared_error

import lunardate 

patient_data = pd.read_csv("../input/patient-volume/abc.csv",index_col='date',parse_dates=['date'])

patient_data.head()



weather_data = pd.read_csv("../input/wheather-dataset/Weater_exchange.csv",index_col='expire_time_gmt', parse_dates=['expire_time_gmt'])



weather_data.head()
# patient_data = patient_data.dropna()

patient = patient_data.index.value_counts()

patient = patient.resample("1D").mean()

patient.isnull().sum()
patient_null = patient[patient.isnull()]

patient_null.index.dayofweek.value_counts()

null = pd.DataFrame({"calendar":list(patient_null.index.dayofweek)}, index = patient_null.index)

null[null.values != 6]
# Tinh nhiet do trung binh moi ngay.

weather = weather_data.temp.resample('1D').max()

# Count volume patient everyday



# Tao  data frame chua 2 series tren

frame = pd.DataFrame({"volume": patient,"temp": weather})

frame = frame.fillna(0)

frame.head()





holiday = [(28,12),(29,12),(30,12),(31,12),(1,1),(2,1),(3,1),(4,1),(5,1),(6,1),(7,1),(8,1)]

holiday1 = [(30,4),(1,5),(2,9)]
lunar = [lunardate.LunarDate.fromSolarDate(i.year,i.month,i.day) for i in list(frame.index)]



frame["lunar"] = pd.Series(lunar,index = frame.index)

frame["date"]= frame.index

frame["holiday"] = frame.apply(lambda x: 1 if ((x.lunar.day,x.lunar.month) in holiday) or (x.date.weekday()==6) or ((x.date.day,x.date.month) in holiday1) else 0, axis = 1)

# frame["holiday"]= pd.Series([1 if (i.lunar.day,i.lunar.month) in holiday else 0 for i in frame], index = frame.index)

frame.head()


# resample freq = week

frame_weekly = pd.DataFrame({"volume":frame.volume.resample("1W").sum(),"holiday":frame.holiday.resample("1W").sum(),"temp":frame.temp.resample("1W").max()})

frame_weekly.describe()
import calendar

frame_month = pd.DataFrame({"volume":frame.volume.resample("1M").sum(),"holiday":frame.holiday.resample("1M").sum(),"temp": frame.temp.resample("1M").max()})



frame_month["totaldayinmonth"] = pd.Series([calendar.monthrange(x.year,x.month)[1] for x in frame_month.index],index = frame_month.index)

frame_month["dayinmonth"] = frame_month.totaldayinmonth - frame_month.holiday

frame_month
# Creating a Timestamp

timestamp = pd.Timestamp(2017, 1, 1, 12)

timestamp
# Creating a period

period = pd.Period('2017-01-01')

period
# Checking if the given timestamp exists in the given period

period.start_time < timestamp < period.end_time
# Converting timestamp to period

new_period = timestamp.to_period(freq='H')

new_period
# Converting period to timestamp

new_timestamp = period.to_timestamp(freq='H', how='start')

new_timestamp
# Creating a datetimeindex with daily frequency

dr1 = pd.date_range(start='1/1/18', end='1/9/18')

dr1
df = pd.DataFrame({'year': [2015, 2016], 'month': [2, 3], 'day': [4, 5]})

df
df = pd.to_datetime(df)

df
df = pd.to_datetime('01-01-2017')

df
frame_month.volume[:50].plot()

shifted = frame_month.volume[:50].shift(12).plot()

shifted.legend(['Volume','Volume Lag'])

plt.figure(figsize=(16,6))

plt.show()
frame_month.volume.plot()

plt.title('Volume patient')

plt.figure(figsize=(16,6))

plt.show()

frame_weekly.describe()
frame_weekly.info()
frame_month.hist(bins=100)
frame_weekly.hist(bins=100)
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

plt.figure(figsize=(16,6))

plot_acf(frame_month.volume,lags=36,title="Patient Volume")



plt.show()
plt.figure(figsize=(16,6))

plot_pacf(frame_month.volume,lags=48)



plt.show()
plt.figure(figsize=(16,6))

plot_pacf(frame_weekly.volume,lags=36)



plt.show()
model = ARMA(frame_month.volume[:len(frame_month)-12], order=(3,0))

result = model.fit()

result.save('../input/model.pkl')

print(result.summary())

print("μ={} ,ϕ={}".format(result.params[0],result.params[1]))

print(result.params)

# Predicting simulated AR(1) model 

result.plot_predict(start=1, end=len(frame_month))

plt.figure(figsize=(16,6))

plt.show()
rmse = math.sqrt(mean_squared_error(frame_month.volume[len(frame_month)-12:], result.predict(start=len(frame_month)-12,end=len(frame_month)-1)))

print("The root mean squared error is {}.".format(rmse))
frame_month.corr()
from sklearn.preprocessing import Normalizer

# retrain with exog

t = frame_month.holiday

i = frame_month.temp





f = pd.DataFrame({"holiday":t,"volume":frame_month.volume,"temp":i})

f = f/f.max()



# retrain with exog

ar = []

for i in range(len(frame_month.volume)-12,len(frame_month.volume)):

    train = frame_month.volume[:i]

    test = frame_month.volume.iloc[i]

    fit1 = ARMA(train, order=(3,0),exog = np.asarray(frame_month.dayinmonth[:len(train)])).fit()

    

#     y = fit1.forecast(step = 1)

    y= fit1.predict(start=len(train),end=len(train),exog = frame_month.dayinmonth.iloc[len(train)])

    ar.append(abs(test-y[0]))

    

print("MAE:",sum(ar)/12)
ar


ar1 = []



for i in range(len(frame_month.volume)-12,len(frame_month.volume)):

    train = frame_month.volume[:i]

    test = frame_month.volume.iloc[i]

    fit1 = ARMA(train, order=(3,0),exog = f[["holiday",'temp']][:len(train)].values).fit()

    fit1.params

    y= fit1.predict(start=len(train),end=len(train),exog = f[["holiday",'temp']].iloc[len(train)])

    ar1.append(abs(test-y[0]))

    

print("MAE:",sum(ar1)/12)
diff_volume = frame_month.volume[:len(frame_month)-12].diff()

diff_volume.head()
frame_month.volume.head()
diff_volume.plot()

plt.figure(figsize=(16,6))

plt.show()
# Predicting humidity level of Montreal

humid = ARMA(frame_month.volume[:len(frame_month)-12].diff().iloc[1:].values, order=(3,0))

res = humid.fit()

print(res.summary())

res.plot_predict(start=0, end=len(frame_month)-12)

plt.figure(figsize=(16,6))

plt.show()
model = ARMA(frame_month.volume[:len(frame_month)-12], order=(0,1))

result = model.fit()



print(result.summary())

print("μ={} ,ϕ={}".format(result.params[0],result.params[1]))

print(result.params)
# Predicting simulated AR(1) model 

result.plot_predict(start=1, end=len(frame_month))

plt.figure(figsize=(16,6))

plt.show()
modelARMA = ARMA(frame_month.volume[:len(frame_month)-12], order=(3,2))

resultARMA  = modelARMA.fit()

print(resultARMA .summary())

print("μ={} ,ϕ={}".format(resultARMA .params[0],resultARMA .params[1]))
# Predicting simulated AR(1) model 

rcParams['figure.figsize'] = 16, 6

resultARMA.plot_predict(start=1, end=len(frame_month))

plt.show()
rmseARMA = math.sqrt(mean_squared_error(frame_month.volume[len(frame_month)-12:], resultARMA.predict(start=len(frame_month)-12,end=len(frame_month)-1)))

print("The root mean squared error is {}.".format(rmseARMA))
rcParams['figure.figsize'] = 16, 6

model = ARIMA(frame_month.volume[:len(frame_month)-12], order=(3,1,0))

resultARIMA = model.fit()

print(resultARIMA.summary())

resultARIMA.plot_predict(start=1, end=len(frame_month))

plt.show()



rmseARIMA = math.sqrt(mean_squared_error(frame_month.volume[len(frame_month)-12:], resultARIMA.predict(start=len(frame_month)-12,end=len(frame_month)-1,typ='levels')))

print("The root mean squared error is {}.".format(rmseARIMA))

from statsmodels.tsa.api import ExponentialSmoothing
# print MAE predict and AIC of model

z = []

# for i in range(len(frame_month.volume)-12,len(frame_month.volume)):

#     train = frame_month.volume[:i]

#     test = frame_month.volume.iloc[i]

#     fit1 =ExponentialSmoothing(np.asarray(train),trend = "add", seasonal  = "add" ,damped  = True).fit()



#     y= fit1.forecast(1)

#     z.append(abs(test-y[0]))

#     print(abs(test-y[0]), fit1.aic)

for i in range(len(frame_month.volume)-36,len(frame_month.volume)):

    train = frame_month.volume[:i]

    test = frame_month.volume.iloc[i]

    fit1 =ExponentialSmoothing(np.asarray(train) ,seasonal_periods=12, trend='additive', damped=True, seasonal='add').fit()

    y= fit1.forecast(1)

    z.append(abs(test-y[0]))

    print(abs(test-y[0]), frame_month.volume[i:i+1])

print("MAE:",sum(z)/36)
ets = pd.Series(z)

ets.describe()
# z = []

# for i in range(len(frame_month.volume)-12,len(frame_month.volume)):

#     train = frame_month.volume[:i]

#     test = frame_month.volume.iloc[i]

#     fit1 =ExponentialSmoothing(np.asarray(train)).fit()



#     y= fit1.forecast(1)

#     z.append(abs(test-y[0]))

#     print(abs(test-y[0]), fit1.aic)

# print("MAE:",sum(z)/12)
ExponentialSmoothing =ExponentialSmoothing(np.asarray(frame_month.volume[:len(frame_month.volume)-12]) )

resultExponentialSmoothing = ExponentialSmoothing.fit()

print(resultExponentialSmoothing.aic)

y_hat= resultExponentialSmoothing.forecast(12)

print(abs(y_hat - frame_month.volume[len(frame_month.volume)-12:]).sum()/12)
# univariate lstm example

from numpy import array

from keras.models import Sequential

from keras.layers import LSTM

from keras.layers import Dense

 

# split a univariate sequence into samples

def split_sequence(sequence, n_steps, hol):

	X, y = list(), list()

	for i in range(len(sequence)):

		# find the end of this pattern

		end_ix = i + n_steps

		# check if we are beyond the sequence

		if end_ix > len(sequence)-1:

			break

		# gather input and output parts of the pattern

		seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]

		seq_x  = np.append(seq_x,hol[end_ix])       

		X.append(seq_x)

		y.append(seq_y)

	return array(X), array(y)

l = len(frame_month) 

# define input sequence

raw_seq = frame_month.volume[:l-12].values

hol = frame_month.holiday[:l-12].values

# choose a number of time steps

n_steps = 3

# split into samples

X, y = split_sequence(raw_seq, n_steps,hol)

# reshape from [samples, timesteps] into [samples, timesteps, features]

n_features = 1

X = X.reshape((X.shape[0], X.shape[1], n_features))



# define model

model = Sequential()

model.add(LSTM(200, activation='relu', input_shape=(4, n_features)))

model.add(Dense(1))

model.compile(optimizer='adam', loss='mse')

# fit model

model.fit(X, y, epochs=1000, verbose=0)

# demonstrate prediction

point = l-12

xtest = list(frame_month.volume[point-3:point])

xtest.append(frame_month.holiday[point])



x_input = array(xtest)

x_input = x_input.reshape((1, 4, n_features))

yhat = model.predict(x_input, verbose=0)

print(yhat)

print("Except: ",frame_month.volume[l-12])
!pip install pmdarima

import pmdarima as pm
pmsarima_exog = []

for i in range(1,24):

    train = frame_month.volume[:len(frame_month.volume)-i]

    test = frame_month.volume.iloc[len(frame_month.volume)-i]

    smodel = pm.auto_arima(frame_month.volume[:len(frame_month.volume)-i], start_p=1, start_q=1, trend = "ct",exogenous  = np.asarray(frame_month[["dayinmonth"]][:len(frame_month.volume)-i]),

                         test='adf',

                         max_p=3, max_q=3, m=12,

                         start_P=0,start_Q = 0, seasonal=True,

                         D=1,

                         trace=True,

                         error_action='ignore',

                         suppress_warnings=True, 

                         stepwise=True)

    

    y= smodel.predict(n_periods = 1,exogenous  = np.asarray(frame_month[["dayinmonth"]][len(frame_month.volume)-i:len(frame_month.volume)-i+1]))

    

    pmsarima_exog.append(abs(test-y[0]))

    

print("MAE:",sum(pmsarima_exog)/24)
pmsarima_exog

pma = pd.Series(pmsarima_exog)

pma.describe()