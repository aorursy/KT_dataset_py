# required pacakges to be imported

import pandas as pd

import matplotlib as plt

import numpy as np

from pandas import DataFrame

from datetime import date

from matplotlib import style

import matplotlib.pylab as plt

%matplotlib inline

from matplotlib.pylab import rcParams

rcParams['figure.figsize']=10,6

import math

# importing the dataset from local storage

#data = pd.read_csv("C:\\Users\\KANDIRAJU\\Downloads\\time-series-datasets\\Electric_Production.csv")

#data.head()

data = pd.read_csv("../input/Electric_Production.csv")

data.head()
# renaming the column names as per my convenience ( this is optional if you wish to perform) 

data.rename(columns = {'DATE' : 'date', 'Value' : 'value'}, inplace=True)

data.head()
# to plot a graph, index has to be set. it is not possible to plot the graph without index.

data.set_index('date', inplace=True)

data.head()

plt.xlabel("date")

plt.ylabel("value")

plt.title("production graph")

from pylab import rcParams

rcParams['figure.figsize'] = 10,6

plt.plot(data); 

# we will notice that the x axis is messed up, this is because, it plotted all the date points and the numbers got overlapped.
rolmean = data.rolling(window=12).mean()

print(rolmean.head(20))
std = data.rolling(window=12).std()

print(std.head(20))
# plot rolling statistics

original_data = plt.plot(data, color='blue',label='original data')

mean = plt.plot(rolmean,color ='red',label='rolling mean')

std = plt.plot(std,color ='black',label='standard deviation')

plt.title("mean, std & original data")

plt.xlabel("date")

plt.ylabel("value")

plt.legend()

from pylab import rcParams

rcParams['figure.figsize'] = 10,6

plt.show(block =False)
# perform dickey fuller test (ADFT)

from statsmodels.tsa.stattools import adfuller 

adft = adfuller(data['value'],autolag='AIC')

# output for dft will give us without defining what the values are.

#hence we manually write what values does it explains using a for loop

output = pd.Series(adft[0:4],index=['Test Statistics','p-value','No. of lags used','Number of observations used'])

for key,values in adft[4].items():

    output['critical value (%s)'%key] =  values

print(output)
data_logscale = np.log(data)

# logarithmic function is used to scale the data to a certain extent.

plt.plot(data_logscale)

plt.title("Log scale")

plt.xlabel("date")

plt.ylabel("value")

#plt.legend()

from pylab import rcParams

rcParams['figure.figsize'] = 10,6

plt.show(block =False)
#determining the rolling mean(average) for the log data. Perform the same steps which are performed on the data before.

moving_average = data_logscale.rolling(window=12).mean()

#print(rolmean_log)

#determining the standard deviation ( same steps! )

std_dev = data_logscale.rolling(window=12).std()

#print(std_log)

plt.plot(moving_average, color='red')

plt.plot(data_logscale, color='blue')

plt.plot(std_dev, color='black');
movingavg_logscale = data_logscale - moving_average

movingavg_logscale.head(15)
movingavg_logscale.dropna(inplace=True)

movingavg_logscale.head(10)


from statsmodels.tsa.stattools import adfuller 

def test_stationarity(timeseries):

    

    #determining the rolling statistics for timeseries

    

    movingAverage = timeseries.rolling(window=12).mean()

    movingSTD = timeseries.rolling(window=12).std()

    

    #plotting the rolling statistics for timeseries

    

    timeseries_original =plt.plot(timeseries, color='blue',label = 'original graph')

    timeseries_mean =plt.plot(movingAverage, color='red',label = 'movingAverage')

    timeseries_std =plt.plot(movingSTD, color='black',label = 'movingSTD')

    plt.legend(loc='best')

    plt.title("rolling mean & standard deviation of timeseries")

    plt.show(block=False)

    

    #perform dickey fuller test

    

    print("results of dickey fuller test")

    adft = adfuller(data['value'],autolag='AIC')

    

    # output for dft will give us without defining what the values are.

    #hence we manually write what values does it explains using a for loop

    

    output = pd.Series(adft[0:4],index=['Test Statistics','p-value','No. of lags used','Number of observations used'])

    for key,values in adft[4].items():

        output['critical value (%s)'%key] =  values

    print(output)

    
from pylab import rcParams

rcParams['figure.figsize'] = 10,6

# we are going to use the function here.

test_stationarity(movingavg_logscale)
print(data_logscale.head())
weighted_average = data_logscale.ewm(halflife=12, min_periods=0,adjust=True).mean()

print(weighted_average.head())


plt.plot(data_logscale)

plt.plot(weighted_average, color='red')

plt.xlabel("date")

plt.ylabel("value")

from pylab import rcParams

rcParams['figure.figsize'] = 10,6

#plt.legend()

plt.show(block =False)
logScale_weightedMean = data_logscale-weighted_average

# use the same function defined above and pass the object into it.

from pylab import rcParams

rcParams['figure.figsize'] = 10,6

test_stationarity(logScale_weightedMean)
data_log_shift = data_logscale - data_logscale.shift()

plt.xlabel("date")

plt.ylabel("value")

plt.title("shifted timeseries")

from pylab import rcParams

rcParams['figure.figsize'] = 10,6

plt.plot(data_log_shift)
# We are dropping the NaN values, and the data_log_shift value here is 'd'

data_log_shift.dropna(inplace=True)

# using the same fuction call and plotting the graph.

from pylab import rcParams

rcParams['figure.figsize'] = 10,6

test_stationarity(data_log_shift)
# next is to segregate the differentiated values to decompose, we use seasonal decompose method from stats model.

# !pip install statsmodels

# !pip install --upgrade patsy

import statsmodels.api as sm

from statsmodels.tsa.seasonal import seasonal_decompose

#decomposition = seasonal_decompose(data_logscale,model='additive', freq=12).plot()



# plotting the graphs induvidually

decomposition = seasonal_decompose(data_logscale,model='additive', freq=12)

trend = decomposition.trend

seasonality =decomposition.seasonal

# ensure that the residual method is just " resid "

# check the values inside the subplots are 411,412,413,414 which mean, there are 4 graphs in total(1st number in the value)

residual =decomposition.resid

plt.subplot(411)

plt.plot(data_logscale,label= 'original')

plt.legend(loc='best')

plt.plot()

plt.subplot(412)

plt.plot(trend,label= 'trend')

plt.legend(loc='best')

plt.plot()

plt.subplot(413)

plt.plot(seasonality,label= 'seasonality')

plt.legend(loc='best')

plt.plot()

plt.subplot(414)

plt.plot(residual,label= 'residual')

plt.legend(loc='best')

plt.plot()

plt.tight_layout()



decomposed_logdata = residual

decomposed_logdata.dropna(inplace=True)

test_stationarity(decomposed_logdata)

# plot acf  and pacf graphs ( auto corellation function and partially auto corellation function )

# to find 'p' from p,d,q we need to use, PACF graphs and for 'q' use ACF graph

from statsmodels.tsa.stattools import acf,pacf

# we use d value here(data_log_shift)

acf = acf(data_log_shift, nlags=15)

pacf= pacf(data_log_shift, nlags=15,method='ols')



# ols stands for ordinary least squares used to minimise the errors



# 121 and 122 makes the data to look side by size 



#plot PACF

plt.subplot(121)

plt.plot(acf) 

plt.axhline(y=0,linestyle='-',color='blue')

plt.axhline(y=-1.96/np.sqrt(len(data_log_shift)),linestyle='--',color='black')

plt.axhline(y=1.96/np.sqrt(len(data_log_shift)),linestyle='--',color='black')

plt.title('Auto corellation function')

plt.tight_layout()





#plot ACF

plt.subplot(122)

plt.plot(pacf) 

plt.axhline(y=0,linestyle='-',color='blue')

plt.axhline(y=-1.96/np.sqrt(len(data_log_shift)),linestyle='--',color='black')

plt.axhline(y=1.96/np.sqrt(len(data_log_shift)),linestyle='--',color='black')

plt.title('Partially auto corellation function')

plt.tight_layout()



# in order to find the p and q values from the above graphs,

#  we need to check,where the graph cuts off the origin or drops to zero for the first time

# from the above graphs the p and q values arw merely close to 2 where the graph cuts off the orgin ( draw the line to x axis)

# now we have p,d,q values. So now we can substitute in the ARIMA model and lets see the output.
from statsmodels.tsa.arima_model import ARIMA



# calculating the AR model

model = ARIMA(data_logscale, order =(2,1,0))

# consider MA as 0 in MA_model

AR_result = model.fit()

plt.plot(data_log_shift)

plt.plot(AR_result.fittedvalues, color='red')

plt.title("sum of squares of residuals")

print('RSS : %f' %sum((AR_result.fittedvalues-data_log_shift["value"])**2))
# less the RSS more effective the model is
# calculating the MA model

model = ARIMA(data_logscale, order =(2,1,0))

# consider MA as 0 in MA_model

MA_result = model.fit()

plt.plot(data_log_shift)

plt.plot(MA_result.fittedvalues, color='red')

plt.title("sum of squares of residuals")

print('RSS : %f' %sum((MA_result.fittedvalues-data_log_shift["value"])**2))

# There is no need of finding the AR and MA values, this is just for our referrence, we already know the values of p,d,q

# you can simply plot the ARIMA model and check for the results.

# calculating the ARIMA model

model = ARIMA(data_logscale, order =(3,1,3))

ARIMA_result = model.fit()

plt.plot(data_log_shift)

plt.plot(ARIMA_result.fittedvalues, color='red')

plt.title("sum of squares of residuals")

print('RSS : %f' %sum((ARIMA_result.fittedvalues-data_log_shift["value"])**2))
# we founded the predicted values in the above code and we need to print the values in the form of series

ARIMA_predicts = pd.Series(ARIMA_result.fittedvalues,copy=True)

ARIMA_predicts.head()
# finding the cummulative sum

ARIMA_predicts_cumsum = ARIMA_predicts.cumsum()

print(ARIMA_predicts_cumsum.head())
ARIMA_predicts_log = pd.Series(data_logscale['value'],index =data_logscale.index)

ARIMA_predicts_log = ARIMA_predicts_log.add(ARIMA_predicts_cumsum,fill_value=0)

print(ARIMA_predicts_log.head())
# converting back to the exponential form results in getting back to the original data.

ARIMA_final_preditcs = np.exp(ARIMA_predicts_log)

rcParams['figure.figsize']=10,10

plt.plot(data)

plt.plot(ARIMA_predicts_cumsum)
from matplotlib.pylab import rcParams

rcParams['figure.figsize']=10,10

plt.plot(ARIMA_predicts_cumsum)

plt.plot(data)

#future prediction

from matplotlib.pylab import rcParams

rcParams['figure.figsize']=15,10

ARIMA_result.plot_predict(1,500)

x=ARIMA_result.forecast(steps=200)
# from the above graph, we calculated the future predictions till 2024

# the greyed out area is the confidence interval wthe predictions will not cross that area.

# check the predicted values for ARIMA_result.plot_predict(1,500)

ARIMA_result.forecast(steps=200)
# Finally we calculated the units(value) of electricity is consumed in the coming future using time series analysis.
# **********                                 THE END                                                        **************** #