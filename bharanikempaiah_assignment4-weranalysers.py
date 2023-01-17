# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from scipy import interpolate

import matplotlib.pyplot as plt

from statsmodels.tsa.seasonal import seasonal_decompose

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

from statsmodels.tsa.stattools import adfuller

import warnings

import itertools

warnings.filterwarnings("ignore")

import statsmodels.api as sm

import matplotlib



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#Reading datasets

df_train=pd.read_csv("/kaggle/input/daily-climate-time-series-data/DailyDelhiClimateTrain.csv")

df_test=pd.read_csv("/kaggle/input/daily-climate-time-series-data/DailyDelhiClimateTest.csv")

df=df_train.interpolate(method='linear')





#Changing to date type to facilitate indexing

df['conv_date']=pd.to_datetime(df.date)

#(1a)



#Getting the time series ready to plot by indexing via date

df_q1= df.resample('D',on="conv_date").mean() 

plt.figure(figsize=(30,10))

plt.plot(df_q1['meantemp'],'-',label="meantemp")

plt.xlabel('Date')

plt.ylabel('MeanTemp')

plt.title('Plot of MeanTemp')

plt.legend(loc=2)

plt.show()



#(1)(b)(i)



#Performing smoothing average with an average as provided in the question

# In iloc[,1], 1 stands for meantemp

#We also fill the NA values with 0 



df_q1['SMA_10'] = df_q1.iloc[:,0].rolling(window=10).mean()

df_q1=df_q1.fillna(0)

plt.figure(figsize=(30,10))

plt.plot(df_q1['SMA_10'],'-',label="SMA(10)")

plt.xlabel('Date')

plt.ylabel('MeanTemp')

plt.title('Simple moving average')

plt.legend(loc=2)

plt.show()





#(i)(b)(ii)



#Applying weighted average as a dot product i.e multiplication of the data point and the filter weight divided by sum of values(done by the lambda function)

#Here we fill the NA with 0



weights=[1.5,1.5,1,0.5,0.5,0.5,0.5,1,1.5,1.5]

weights=np.asarray(weights)

df_q1['WMA_10'] = df_q1.iloc[:,0].rolling(10).apply(lambda temp: np.dot(temp,weights)/weights.sum(),raw=True)

df_q1=df_q1.fillna(0)

plt.figure(figsize=(30,10))

plt.plot(df_q1['WMA_10'],'-',label="WMA")

plt.xlabel('Date')

plt.ylabel('MeanTemp')

plt.title('Weighted moving average')

plt.legend(loc=2)

plt.show()
plt.figure(figsize=(30,10))

plt.plot(df_q1['meantemp'],'-',label="Original meantemp",color="green",alpha=0.3)

plt.plot(df_q1['SMA_10'],'-',label="SMA(10)",color="blue")

plt.plot(df_q1['WMA_10'],'-',label="WMA",color="red")

plt.xlabel('Date')

plt.ylabel('MeanTemp')

plt.title('Plot of meantemp')

plt.legend()

plt.show()
#(i)(c)(i)



#Since we have daily data resampling to Hourly based on mean will introduce NaNs for 23 hours out of 24 per day. Linear interpolating the values





df_hour = df.resample('H',on="conv_date").mean()

df_h_fin=df_hour.interpolate(method='linear')

plt.figure(figsize=(30,10))

plt.plot(df_h_fin['meantemp'],'-',label="Resampled Hourly")

plt.xlabel('Date')

plt.ylabel('MeanTemp')

plt.title('Hourly resampled data')

plt.legend(loc=2)

plt.show()

#(i)(c)(ii)



#We perform resampling weekly based on date and taking the mean of all observations



df_week = df.resample('W',on="conv_date").mean() 

plt.figure(figsize=(30,10))

plt.plot(df_week['meantemp'],'-',label="Weekly")

plt.xlabel('Date')

plt.ylabel('MeanTemp')

plt.title('Weekly resampled data')

plt.legend(loc=2)

plt.show()
#)(i)(c)(iii)



#Performing resampling monthly based on date by making use of mean of all observations belonging to a month

df_month = df.resample('M',on="conv_date").mean() 

plt.figure(figsize=(30,10))

plt.plot(df_month['meantemp'],'-',label="Monthly")

plt.xlabel('Date')

plt.ylabel('MeanTemp')

plt.title('Monthly resampled data')

plt.legend(loc=2)

plt.show()
#(i)(c)(iv)



#Resampling quarterly based on date and taking mean of all days per quarter

df_quarter = df.resample('Q',on="conv_date").mean() 

plt.figure(figsize=(30,10))

plt.plot(df_quarter['meantemp'],'-',label="Quarterly")

plt.xlabel('Date')

plt.ylabel('MeanTemp')

plt.title('Quarterly resampled data')

plt.legend(loc=2)

plt.show()
#(2)(a)(i)



#Decomposing the given time series to obtain the components

components = seasonal_decompose(df_month['meantemp'])

components.plot()

plt.show()
#To find if the series is additive or multiplicative we add the individual components 

#and see which out of additive or multiplicative can bring back the original data



additive = components.trend + components.seasonal + components.resid

multiplicative = components.trend * components.seasonal * components.resid



#Additive

plt.plot(components.observed, label="Original")

plt.plot(additive, label="Additive")

plt.xlabel('Date')

plt.ylabel('Mean temperature')

plt.title('Additive series')

plt.legend(loc=2)

plt.show()



#Multiplicative

plt.plot(components.observed, label="Original")

plt.plot(multiplicative, label="Multiplicative")

plt.xlabel('Date')

plt.ylabel('Mean temperature')

plt.title('Miltiplicative Series')

plt.legend(loc=2)

plt.show()

#(2)(a)(ii)



#Calling the corresponding ACF and PACF functions



plot_acf(df['meantemp'])

plt.xlabel('Lag')

plt.ylabel('Auto correlations')

plt.title('Plot of ACF')

plt.show()



plot_pacf(df['meantemp'])

plt.xlabel('Lag')

plt.ylabel('Partial Auto correlations')

plt.title('Plot of PACF')

plt.show()



#The Dickey Fuller test is one of the most popular statistical tests for stationarity 

#It can be used to determine the presence of unit root in the series, and hence help understand if the series is stationary or not. 

#The null and alternate hypothesis of this test are:

#Null Hypothesis: The series has a unit root (value of a =1)

#Alternate Hypothesis: The series has no unit root.





dftest = adfuller(df['meantemp'], autolag='AIC')

dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])

for key,value in dftest[4].items():

    dfoutput['Critical Value (%s)'%key] = value

print (dfoutput)



#If the test statistic is less than critical value, reject the null hypothesis (series is stationary). 

#When it is greater, we fail to reject the null hypothesis (series is not stationary).

#Using first order differentials to convert to stationary



df['meantemp_diff'] = df['meantemp'] - df['meantemp'].shift(1)

#Repeating the 0th value for the NA

df['meantemp_diff'][0] = df['meantemp'][0]



#Performing the Dickey Fuller test



dftest = adfuller(df['meantemp_diff'], autolag='AIC')

dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])

for key,value in dftest[4].items():

    dfoutput['Critical Value (%s)'%key] = value

print (dfoutput)



#We know that the order of differencing = 1  

#From the PACF plot we know that p should be = 1 and hence keep it fixed



p=[1]

d=[0,1]

q=[0,1]



min_aic=10000

pdq = list(itertools.product(p, d, q))

seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]



#For possible combinations of (p,d,q) and (P,D,Q) we loop over and try to observe how the model performs

for param in pdq:

    for param_seasonal in seasonal_pdq:

        try:

            mod = sm.tsa.statespace.SARIMAX(df['meantemp'],order=param,seasonal_order=param_seasonal,enforce_stationarity=False,enforce_invertibility=False)

            results = mod.fit()

            

            print('SARIMA{}x{}12 - AIC:{}'.format(param,param_seasonal,results.aic))

            if(results.aic<min_aic):

                min_aic=results.aic

                min_param=param

                min_seasonal=param_seasonal

            

        except: 

            continue



print('min aic = ',min_aic)

print('Parameters for non seasonal(p,d,q) = ',min_param)

print('Parameters for Seasonal(P,D,Q) = ',min_seasonal)
#Fitting the model to the test dataset by passing the order and seasonal order obtained from before



mod = sm.tsa.statespace.SARIMAX(df_test['meantemp'],

                                order=min_param,

                                seasonal_order=min_seasonal,

                                enforce_stationarity=False,

                                enforce_invertibility=False)

results = mod.fit()

print(results.summary().tables[1])
#Predicting

yhat = results.predict(start=0, end=len(df_test['meantemp']))

y_truth = df_test['meantemp']



#Plotting the predicted values on top of the test set



plt.figure(figsize=(30,10))

plt.plot(yhat, label = "Predicted")

plt.plot(y_truth, label = "Actual")

plt.xlabel('Date')

plt.ylabel('meantemp')

plt.title('Plot of meantemp for test set')

plt.legend(loc=2)

plt.show()



#Calculating error

mse = ((yhat - y_truth) ** 2).mean()

print('The Mean Squared Error is {}'.format(round(mse, 2)))

print('The Root Mean Squared Error is {}'.format(round(np.sqrt(mse), 2)))




