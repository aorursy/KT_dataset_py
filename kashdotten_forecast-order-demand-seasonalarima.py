# Input data files are available in the "../input/" directory.

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#Import Libraries

import pandas as pd

import numpy as np

import seaborn as sb



import matplotlib.pyplot as plt

%matplotlib inline



from scipy.stats import norm, skew #for some statistics

from scipy import stats #qqplot

import statsmodels.api as sm #for decomposing the trends, seasonality etc.



from statsmodels.tsa.statespace.sarimax import SARIMAX #the big daddy
#Import the data and parse dates.

df  = pd.read_csv('../input/productdemandforecasting/Historical Product Demand.csv', parse_dates=['Date'])
#Don't judge it by the cover.

df.head(5)
#Check the cardinality.

df.shape
#Check the data types.

df.dtypes
# Check any number of columns with NaN

print(df.isnull().any().sum(), ' / ', len(df.columns))

# Check any number of data points with NaN

print(df.isnull().any(axis=1).sum(), ' / ', len(df))
#Lets check where these nulls are.

print (df.isna().sum())

print ('Null to Dataset Ratio in Dates: ',df.isnull().sum()[3]/df.shape[0]*100)

#There are missing values in Dates.
#Drop na's.



#Since the number of missing values are about 1%, I am taking an 'executive decision' of removing them. ;) 

df.dropna(axis=0, inplace=True) #remove all rows with na's.

df.reset_index(drop=True)

df.sort_values('Date')[10:20] #Some of the values have () in them.
#Target Feature - Order_Demand

#Removing () from the target feature.

df['Order_Demand'] = df['Order_Demand'].str.replace('(',"")

df['Order_Demand'] = df['Order_Demand'].str.replace(')',"")



#Next step is to change the data type.

df['Order_Demand'] = df['Order_Demand'].astype('int64')
#Get the lowest and highest dates in the dataset.

df['Date'].min() , df['Date'].max()

#There is data for 6 years. great.
#Lets start with 2012 and cap it 2016 december. Since the dates before 2012 have a lot of missing values - inspected and checked using basic time series plot.

df = df[(df['Date']>='2012-01-01') & (df['Date']<='2016-12-31')].sort_values('Date', ascending=True)
#Lets check the ditribution of the target variable (Order_Demand)

from matplotlib import rcParams

# figure size in inches

rcParams['figure.figsize'] = 4,2



sb.distplot(df['Order_Demand'], fit=norm)



#Get the QQ-plot

fig = plt.figure()

res = stats.probplot(df['Order_Demand'], plot=plt)

plt.show()
#The data is highly skewed, but since we'll be applying ARIMA, it's fine.

df['Order_Demand'].skew()
#Just in case if there needs to be some transformation, it can be done by either taking log values or using box cox.



## In case you need to normalize data, use Box Cox. Pick the one that looks MOST like a normal distribution.

# for i in [1,2,3,4,5,6,7,8]:

#     plt.hist(df['Order_Demand']**(1/i), bins= 40, normed=False)

#     plt.title("Box Cox transformation: 1/{}". format(str(i)))

#     plt.show()
#Warehouse shipping by orders.

df['Warehouse'].value_counts().sort_values(ascending = False)
#The amount of orders shipped by each warehouse.

df.groupby('Warehouse').sum().sort_values('Order_Demand', ascending = False)

#Warehouse J is clearly shipping most orders. Although S is shipping more quantity within fewer requested orders.
#Product Category.



print (len(df['Product_Category'].value_counts()))



rcParams['figure.figsize'] = 50,14

sb.countplot(df['Product_Category'].sort_values(ascending = True))



#There's a lot of orders on category19.
#Lets check the orders by warehouse.



#Checking with Boxplots

from matplotlib import rcParams

# figure size in inches

rcParams['figure.figsize'] = 16,4

f, axes = plt.subplots(1, 2)

#Regular Data

fig3 = sb.boxplot( df['Warehouse'],df['Order_Demand'], ax = axes[0])

#Data with Log Transformation

fig4 = sb.boxplot( df['Warehouse'], np.log1p(df['Order_Demand']),ax = axes[1])



del fig3, fig4
#Lets check the Orders by Product Category.

rcParams['figure.figsize'] = 50,12

#Taking subset of data temporarily for in memory compute.

df_temp = df.sample(n=20000).reset_index()

fig5 = sb.boxplot( df_temp['Product_Category'].sort_values(),np.log1p(df_temp['Order_Demand']))

del df_temp, fig5
df = df.groupby('Date')['Order_Demand'].sum().reset_index()

#This gives us the total orders placed on each day.
#Index the date

df = df.set_index('Date')

df.index #Lets check the index
#Averages daily sales value for the month, and we are using the start of each month as the timestamp.

y = df['Order_Demand'].resample('MS').mean()
#In case there are Null values, they can be imputed using bfill.

#y = y.fillna(y.bfill())
#Visualizing time series.



y.plot(figsize=(12,5))

plt.show()



#Takeaway: The sales are always low for the beginning of the year and the highest peak in demand every year is in the

#last quarter. The observed trend shows that orders were higher during 2014-2016 then reducing.



#Lets check it by decomposition.
#The best part about time series data and decomposition is that you can break down the data into the following:

#Time Series Decomposition. 

from pylab import rcParams

import statsmodels.api as sm

rcParams['figure.figsize'] = 18, 8

decomposition = sm.tsa.seasonal_decompose(y, model='additive')

fig = decomposition.plot()

plt.show()
#GRID SEARCH for Param Tuning.

#Sample params for seasonal arima. (SARIMAX).



#For each combination of parameters, we fit a new seasonal ARIMA model with the SARIMAX() function 

#from the statsmodels module and assess its overall quality.



import itertools

p = d = q = range(0, 2)

pdq = list(itertools.product(p, d, q))

seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]

print('Examples of parameter combinations for Seasonal ARIMA...')

print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[1]))

print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[2]))

print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[3]))

print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[4]))
#Get the best params for the data. Choose the lowest AIC.



# The Akaike information criterion (AIC) is an estimator of the relative quality of statistical models for a 

# given set of data. 

# AIC measures how well a model fits the data while taking into account the overall complexity of the model.

# Large AIC: Model fits very well using a lot of features.

# Small AIC: Model fits similar fit but using lesser features. 

# Hence LOWER THE AIC, the better it is.



#The code tests the given params using sarimax and outputs the AIC scores.



for param in pdq:

    for param_seasonal in seasonal_pdq:

        try:

            mod = sm.tsa.statespace.SARIMAX(y,

                                            order=param,

                                            seasonal_order=param_seasonal,

                                            enforce_stationarity=False,

                                            enforce_invertibility=False)



            results = mod.fit()



            print('SARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, results.aic))

        except:

            continue
#Fit the model with the best params.

#ARIMA(1, 1, 1)x(1, 1, 0, 12)12 - AIC:960.5164122018646





#The above output suggests that ARIMA(1, 1, 1)x(1, 1, 0, 12) yields the lowest AIC value: 960.516

#Therefore we should consider this to be optimal option.



from statsmodels.tsa.statespace.sarimax import SARIMAX

mod = sm.tsa.statespace.SARIMAX(y,

                                order=(1, 1, 1),

                                seasonal_order=(1, 1, 0, 12),

                                enforce_stationarity=False,

                                enforce_invertibility=False)

results = mod.fit()

print(results.summary().tables[1])
#Plotting the diagnostics.



#The plot_diagnostics object allows us to quickly generate model diagnostics and investigate for any unusual behavior.

results.plot_diagnostics(figsize=(16, 8))

plt.show()



#What to look for?

#1. Residuals SHOULD be Normally Distributed ; Check

#Top Right: The (orange colored) KDE line should be closely matched with green colored N(0,1) line. This is the standard notation

#for normal distribution with mean 0 and sd 1.

#Bottom Left: The qq plot shows the ordered distribution of residuals (blue dots) follows the linear trend of the samples 

#taken from a standard normal distribution with N(0, 1). 



#2. #Residuals are not correlated; Check

#Top Left: The standard residuals don’t display any obvious seasonality and appear to be white noise. 

#Bottom Right: The autocorrelation (i.e. correlogram) plot on the bottom right, which shows that the time series residuals have 

#low correlation with its own lagged versions.

#Lets get the predictions and confidence interval for those predictions.

#Get the predictions. The forecasts start from the 1st of Jan 2017 but the previous line shows how it fits to the data.

pred = results.get_prediction(start=pd.to_datetime('2014-05-01'), dynamic=False) #false is when using the entire history.

#Confidence interval.

pred_ci = pred.conf_int()



#Plotting real and forecasted values.

ax = y['2013':].plot(label='observed')

pred.predicted_mean.plot(ax=ax, label='One-step ahead Forecast', alpha=.7, figsize=(14, 7))

ax.fill_between(pred_ci.index,

                pred_ci.iloc[:, 0],

                pred_ci.iloc[:, 1], color='blue', alpha=.2)

ax.set_xlabel('Date')

ax.set_ylabel('Order_Demand')

plt.legend()

plt.show()



#Takeaway: The forecats seems to be fitting well to the data. The Blue/purple thicker plot shows the confidence level in the forecasts. 
#Getting the mean squared error (average error of forecasts).

y_forecasted = pred.predicted_mean

y_truth = y['2016-01-01':]

mse = ((y_forecasted - y_truth) ** 2).mean()

print('MSE {}'.format(round(mse, 2)))



#Smaller the better.
print('RMSE: {}'.format(round(np.sqrt(mse), 2)))
#The time can be changed using steps.

pred_uc = results.get_forecast(steps=50)

pred_ci = pred_uc.conf_int()

ax = y.plot(label='observed', figsize=(14, 7))

pred_uc.predicted_mean.plot(ax=ax, label='Forecast')

ax.fill_between(pred_ci.index,

                pred_ci.iloc[:, 0],

                pred_ci.iloc[:, 1], color='k', alpha=.25)

ax.set_xlabel('Date')

ax.set_ylabel('Order_Demand')

plt.legend()

plt.show()



#Far out values are naturally more prone to variance. The grey area is the confidence we have in the predictions.
#And that is how you build a somewhat accurate forecast of the demand. Next steps include checking and modeling demand by category type.