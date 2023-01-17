# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
%pylab inline 



from pylab import rcParams

plt.style.use('fivethirtyeight')

import warnings

warnings.filterwarnings('ignore')
Climate_Data = pd.read_csv('/kaggle/input/daily-climate-time-series-data/DailyDelhiClimateTrain.csv',

                           index_col='date', parse_dates=['date'])
Climate_Data.tail(3)
Climate_Data.describe()
# get the column name with na_values.

col_with_na = [col for col in Climate_Data.columns if Climate_Data[col].isnull().any()]



print(col_with_na)
Climate_Data["humidity"].asfreq('M').plot() 

# asfreq method is used to convert a time series to a specified frequency. Here it is monthly frequency.

plt.title('Humidity in Delhi City over time(Monthly frequency)')

plt.show()
Climate_Data.asfreq('M').plot(subplots=True, figsize=(10,12))

plt.title('Climate in Delhi City from 2013 to 2017')

plt.savefig('climate.png')

plt.show()
Climate_Data.plot(subplots=True, figsize=(10,12))

plt.title('Climate in Delhi City from 2013 to 2017')

plt.savefig('climate.png')

plt.show()
# creating a timestamp

timestamp = pd.Timestamp(2014, 1, 1, 12)



timestamp
# creating a period

period = pd.Period('2014', freq='M')



period
print(period.start_time, '\n', period.end_time)
# Checking if the given timestamp exists in the given period

period.start_time < timestamp < period.end_time
df = pd.DataFrame({'year': [2015, 2016], 'month': [2, 3], 'day': [4, 5], 'temperature':[24, 26]})

df
df['date'] = pd.to_datetime(df[['year','month','day']])

# change the date as the index of dataframe.

df.index = df['date']

df = df['temperature']

df
Climate_Data['2014':]["humidity"].asfreq('M').plot(legend=True)

shifted = Climate_Data["humidity"].asfreq('M').shift(12).plot(legend=True)

shifted.legend(['humdity', 'humdity_shifted'])

plt.show
# we downsample from daily to weekly frequency aggregated using mean.

climate_down_sample = Climate_Data.resample('7D').mean()



climate_down_sample
climate_up_sample = climate_down_sample.resample('D').pad()

climate_up_sample.head(10)
# Step one : Upsampling the Climate Data ('D' -> 'W')

Climate_Data_Weekly = Climate_Data.resample('7D').mean()



# Step two : calculate the changes 

Climate_Data_Weekly['mean_temp_change'] = Climate_Data_Weekly.meantemp.div(Climate_Data_Weekly.meantemp.shift())



# Step three : draw the diff graph. 

Climate_Data_Weekly['mean_temp_change'].plot(figsize=(20, 8))

plt.title('Weekly meantemp change rate')
Climate_Data_Weekly.head(10)
Climate_Data_Weekly['temp_diff'] = Climate_Data_Weekly.meantemp.sub(Climate_Data_Weekly.meantemp.shift())

Climate_Data_Weekly['temp_diff'].plot(figsize=(20, 8))
# in fact, there is a built-in mmethod to do the above job, is the same as the above graph.

Climate_Data_Weekly.meantemp.diff().plot(figsize=(20, 8), color='r')
# Normalizing and comparision

normalized_meantemp = Climate_Data.meantemp.div(Climate_Data.meantemp.iloc[0]).mul(100)

normalized_meanpressure = Climate_Data.meanpressure.div(Climate_Data.meanpressure.iloc[0]).mul(100)

normalized_humidity = Climate_Data.humidity.div(Climate_Data.humidity.iloc[0]).mul(100)

# normalized_wind_speed = Climate_Data.wind_speed.div(Climate_Data.wind_speed.iloc[0]+0.01).mul(100)



# plotting 

rcParams['figure.figsize'] = (16, 6)

normalized_meantemp.plot()

normalized_meanpressure.plot()

normalized_humidity.plot()

# normalized_wind_speed.plot()

plt.legend(['meantemp', 'meanpressure', 'humidity', 'wind_speed'])

plt.show()
# Rolling window functions

rolling_meantemp = Climate_Data.meantemp.rolling('20D').mean()

Climate_Data.meantemp.plot()

rolling_meantemp.plot()

plt.legend(['meantemp', 'Rolling Mean'])



# plotting a rolling mean of 20 day window with original meantemp attribute
# Expandinng window funcitons 

mean_humility = Climate_Data.humidity.expanding().mean()

std_humility = Climate_Data.humidity.expanding().std()



Climate_Data.humidity.plot()

mean_humility.plot()

std_humility.plot()



plt.legend(['humility', 'Expanding mean', 'Expanding std'])

plt.show()
from statsmodels.graphics.tsaplots import plot_acf



# autocorrelation of humility 

plot_acf(Climate_Data.humidity, lags=30, title='humidity')

plt.show()

# in fact, we can only find that the smaller the time interval, the greater the correlation. :)
from statsmodels.graphics.tsaplots import plot_pacf

rcParams['figure.figsize'] = (10, 6)

plot_pacf(Climate_Data.humidity, lags=30)

plt.show()
Climate_Data.meantemp.plot(figsize=(10, 4))
import statsmodels.api as sm



# now, for decomposition

rcParams['figure.figsize'] = 11, 9

decomposed_meantemp_volume = sm.tsa.seasonal_decompose(Climate_Data.meantemp, freq=360) # the frequency is annual 

figure = decomposed_meantemp_volume.plot()

plt.show()
# plotting white noise

rcParams['figure.figsize'] = 16, 6

white_noise = np.random.normal(loc=0, scale=1, size=1000) # loc is mean, scale is variance



plt.plot(white_noise)
# plotting autocorrelation of white noise

plot_acf(white_noise, lags=30, title='autocorrelation of white noise')

plt.show()
# augmented Dickey-Fuller test on volume of meantemp

from statsmodels.tsa.stattools import adfuller

adf = adfuller(Climate_Data.meantemp)

print(f"p-value of meantemp_daily : {float(adf[1])}")

adf = adf = adfuller(Climate_Data_Weekly.meantemp)

print(f"p-value of meantemp_weekly : {float(adf[1])}")
# The original non-stationary plot

decomposed_meantemp_volume.trend.plot()
# the new stationary plot

decomposed_meantemp_volume.trend.diff().plot()
from statsmodels.tsa.arima_process import ArmaProcess



# AR(1) MA(1) model : AR parameter = +0.9

rcParams['figure.figsize'] = 16, 12

plt.subplot(4, 1, 1)

ar1 = np.array([1, -0.9]) # we choose -0.9 as AR parameter is +0.9

ma1 = np.array([1])

AR1 = ArmaProcess(ar1, ma1)

sim1 = AR1.generate_sample(nsample=1000)

plt.title('AR(1) model: AR parameter = +0.9')

plt.plot(sim1)



# We will take care of MA model later

# AR(1) MA(1) AR parameter = -0.9

plt.subplot(4,1,2)

ar2 = np.array([1, 0.9]) # We choose +0.9 as AR parameter is -0.9

ma2 = np.array([1])

AR2 = ArmaProcess(ar2, ma2)

sim2 = AR2.generate_sample(nsample=1000)

plt.title('AR(1) model: AR parameter = -0.9')

plt.plot(sim2)



# AR(2) MA(1) AR parameter = 0.9

plt.subplot(4,1,3)

ar3 = np.array([2, -0.9]) # We choose -0.9 as AR parameter is +0.9

ma3 = np.array([1])

AR3 = ArmaProcess(ar3, ma3)

sim3 = AR3.generate_sample(nsample=1000)

plt.title('AR(2) model: AR parameter = +0.9')

plt.plot(sim3)



# AR(2) MA(1) AR parameter = -0.9

plt.subplot(4,1,4)

ar4 = np.array([2, 0.9]) # We choose +0.9 as AR parameter is -0.9

ma4 = np.array([1])

AR4 = ArmaProcess(ar4, ma4)

sim4 = AR4.generate_sample(nsample=1000)

plt.title('AR(2) model: AR parameter = -0.9')

plt.plot(sim4)

plt.show()
from statsmodels.tsa.arima_model import ARMA



model = ARMA(sim1, order=(1, 0))

result = model.fit()

print(result.summary())

print(f"μ={result.params[0]} ,ϕ={result.params[1]}")
import math

from sklearn.metrics import mean_squared_error



result.plot_predict(start=900, end=1010)

plt.show()



# calculate the mean_squared_error.

rmse = math.sqrt(mean_squared_error(sim1[900:1010], result.predict(start=900, end=999)))

print(f"The root mean squared error is {rmse}.")
Climate_test = pd.read_csv('/kaggle/input/daily-climate-time-series-data/DailyDelhiClimateTest.csv', 

                          index_col='date', parse_dates=['date'])

Climate_test.head()
# get the ground truth 

Climate_test_Weekly = Climate_test.resample('7D').mean()



Climate_test_Weekly.meantemp.plot(figsize=(10, 6))
Climate_Data_Weekly.shape
from statsmodels.tsa.arima_model import ARMA



meantemp_model = ARMA(Climate_Data_Weekly['meantemp'], order=(10, 1))



# fitting our training data.

result = meantemp_model.fit()



# The AIC criterion, also known as the Akaike message criterion.

# is a measure of how well a statistical model fits. The smaller the value, the better the model fits.

print(f'AIC: {result.aic : 0.2f}')



predicted = result.predict('2017', '2018')
result.plot_predict(start=160, end=230)

Climate_test_Weekly.meantemp.plot()

plt.show()
from statsmodels.tsa.arima_model import ARIMA

rcParams['figure.figsize'] = 16, 6

ARIMA_model = ARIMA(Climate_Data_Weekly.meantemp, order=(4, 2, 3))

result = ARIMA_model.fit()

print(result.summary())
result.plot_predict(start=100, end=220)

Climate_test_Weekly.meantemp.plot()

plt.show()
# so we are going to use 2013-1-1 to 2013-5-31 temperatue data to predict 

# 2013-6 temperatue, knowing that temperature's trend is growing during this poried.



train_data = Climate_Data.meantemp[:'2013-4-30']



test_data = Climate_Data.meantemp['2013-5-1':'2013-5-31']



ARIMA_model = ARIMA(train_data, order=(5,2,5))



result = ARIMA_model.fit()

print(result.summary())
result.plot_predict(start=10, end=160)

test_data.plot()

plt.legend(['train_serie', 'prediction', 'ground_truth'])

plt.show()