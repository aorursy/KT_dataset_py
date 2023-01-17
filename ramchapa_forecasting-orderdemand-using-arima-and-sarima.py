# Input data files are available in the Kaggle "../input/" directory.

import os

for dirname,_,filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print (os.path.join(dirname,filename))
import pandas as pd # Data handling and managing

import numpy as np  # Handiling linear Algera

import seaborn as sn

import matplotlib.pyplot as plt

%matplotlib inline





df = pd.read_csv('../input/productdemandforecasting/Historical Product Demand.csv', parse_dates=['Date'])

df.head(100) # Getting the first 100 rows to view the records

#df.shape

#check for all the date types and nature.

df.dtypes
# Check for the columns which got has the NaN values

print(df.isnull().any().sum(), ' / ', len(df.columns))

# Check any number of data points with NaN

print(df.isnull().any(axis=1).sum(),'/', len(df))

#print(df.isnull().any(axis=1).sum(), ' / ', len(df))
#Lets check which column has null values.

print (df.isna().sum())



#Print the Null Value to Dataset Ratio for the column obtained in the above line of code

print ('Null to Dataset Ratio for "Dates" Column '': ',df.isnull().sum()[3]/df.shape[0]*100)

#So, its an clear indcation that There are missing values in Dates and the ratio is 1 %.
#Since the number of missing values are about 1%, So i will be removing them for cleaner workble data. 

#df.dropna(axis=0, inplace=True) #remove all rows with na's.

#df.reset_index(drop=True)

#df.sort_values('Date')[10:20] #Some of the values have () in them.



df.dropna(axis=0, inplace=True) #Remove all the rows with na's

df.reset_index(drop=True)

df.sort_values('Date')[1:50]
#We can notice Some of the values have () in them for "Order_Demand" column, which have to remove.

#Removing () from the "Order_Demand" Column

df['Order_Demand']=df['Order_Demand'].str.replace('(',"")

df['Order_Demand']=df['Order_Demand'].str.replace(')',"")

df.head(100)

#Since the "()" has been removed , Now i Will change the data type.



df['Order_Demand'] = df['Order_Demand'].astype('int64')

#Get the Hieghest and lowest dates in the dataset.

df['Date'].min() , df['Date'].max()

#There is data for 6 years.
from scipy.stats import norm, skew #Import Norm and skew for some statistics

from scipy import stats #Import stats

import statsmodels.api as sm #for decomposing the trends, seasonality etc.



from statsmodels.tsa.statespace.sarimax import SARIMAX #for the Seasonal Forecast





#Lets check the ditribution of the target variable (Order_Demand)

from matplotlib import rcParams

# figure size in inches

rcParams['figure.figsize'] = 10,5



sn.distplot(df['Order_Demand'],fit=norm)



#Get the QQ-plot

fig = plt.figure()

res = stats.probplot(df['Order_Demand'], plot=plt)

plt.show()


## In case we need Data Normilization, We can use Log Values or use Box Cox. Pick the one that looks MOST like a normal distribution.

#for i in [1,2,3,4,5,6,7,8]:

 #   plt.hist(df['Order_Demand']**(1/i), bins= 40, normed=False)

  #  plt.title("Box Cox transformation: 1/{}". format(str(i)))

   # plt.show()
#Considering Warehouse, Product Category columns for UniVariate Analysis.

df['Warehouse'].value_counts().sort_values(ascending=False)
#Now I will get the amount of orders shipped by each warehouse.

df.groupby('Warehouse').sum().sort_values('Order_Demand', ascending = False)

#Warehouse J is clearly shipping most orders. Although S is shipping more quantity within fewer requested orders.
#Product Category analysis

print(len(df['Product_Category'].value_counts()))

rcParams['figure.figsize']=50,14

#sn.countplot(df['Product_Category'].sort_values(ascending=True))
#Creating a Bivariate Analysis for WH and PC with Order Demand as Target Variable.



#Step-01: Check the Order Demand Qty by WareHouse

from matplotlib import rcParams



rcParams['figure.figsize']=20,5 #Figure Size in Inches for Plotting

f, axes = plt.subplots(1,2)



regDataWH=sn.boxplot(df['Warehouse'],df['Order_Demand'],ax=axes[0]) #Create a variable for Regular Data for WH and OD 



logDataWH=sn.boxplot(df['Warehouse'],np.log1p(df['Order_Demand']),ax=axes[1]) #Craete a Variable with Log Transformation



del regDataWH, logDataWH



#Step-02: Check the Order Demand Qty by Product Category (PC)

rcParams['figure.figsize']=20,5

f,axes =plt.subplots(1,2)



regDataPC=sn.boxplot(df['Product_Category'],df['Order_Demand'],ax=axes[0])

logDataPC=sn.boxplot(df['Product_Category'],df['Order_Demand'],ax=axes[1])



del regDataPC, logDataPC

#Exploring the Data as TIME SERIES

#Step-01: Lets calculate the Total  Order Qty placed on by Each Day

df=df.groupby('Date')['Order_Demand'].sum().reset_index()

#Step-02: Indexing the Date Column as for further procssing.

df = df.set_index('Date')

df.index #Lets check the index

#Step-03:#Averages daily sales value for the month, and we are using the start of each month as the timestamp.

monthly_avg_sales = df['Order_Demand'].resample('MS').mean()

#In case there are Null values, they can be imputed using bfill.

monthly_avg_sales = monthly_avg_sales.fillna(monthly_avg_sales.bfill())

#Visualizing time series.



monthly_avg_sales.plot(figsize=(20,10))

plt.show()



#Findings: The sales are always low for the beginning of the year and the highest peak in demand every year is in the

#last quarter. The observed trend shows that orders were higher during 2014-2016 then reducing down slowly.
#Calculate the Seasonality , Trend and Residuals with Decomposition Analysis.



#Using Time Series for Decomposition. 

from pylab import rcParams

import statsmodels.api as sm

rcParams['figure.figsize'] = 20, 10

decomposition = sm.tsa.seasonal_decompose(monthly_avg_sales, model='additive')

fig = decomposition.plot()

plt.show()
#ARIMA



#An ARIMA model is characterized by 3 terms: p, d, q where these three parameters account for seasonality (p), trend (d), and noise in data (q):



#p is the order of the AR term (number of lags of Y to be used as predictors). If it rained for the last week, it is likely it will rain tomorrow.

#q is the order of the MA term (moving average).

#d is the number of differencing required to make the time series stationary. if already stationary d=0.

#But when dealing with SEASONALITY, it is best to incorporate it as 's'. ARIMA(p,d,q)(P,D,Q)s. Where 'pdq' are non seasonal params and 's' is the perdiocity of the time series. 4:quarter, 12:yearly etc.

#If a time series, has seasonal patterns, then you need to add seasonal terms and it becomes SARIMA, short for ‘Seasonal ARIMA’.
#Grid Search and Random Search



#Since ARIMA has hyper params that can be tuned, the objective here is to find the best params using Grid Search.



#GRID SEARCH for Param Tuning.

#Sample params for seasonal arima. (SARIMAX).



#STEP-01:

#For each combination of parameters, we fit a new seasonal ARIMA model with the SARIMAX() function 

#from the statsmodels module and assess its overall quality.



import itertools

p = d = q = range(0, 2)

pdq = list(itertools.product(p, d, q))

seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]

#print('Examples of parameter combinations for Seasonal ARIMA...')

print('SARIMAX1: {} x {}'.format(pdq[1], seasonal_pdq[1]))

print('SARIMAX2: {} x {}'.format(pdq[1], seasonal_pdq[2]))

print('SARIMAX3: {} x {}'.format(pdq[2], seasonal_pdq[3]))

print('SARIMAX4: {} x {}'.format(pdq[2], seasonal_pdq[4]))



#STEP-02:

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

            mod = sm.tsa.statespace.SARIMAX(monthly_avg_sales,

                                            order=param,

                                            seasonal_order=param_seasonal,enforce_stationarity=False,

                                            enforce_invertibility=False)

            results = mod.fit()

            print('SARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, results.aic))

        except:

            continue
#Fit the model with the best params.

#SARIMA(1, 1, 1)x(0, 1, 1, 12)12 - AIC:1351.1631068717465





#The above output suggests that ARIMA(1, 1, 1)x(0, 1, 1, 12)12 yields the lowest AIC value: 1351.1631068717465

#Therefore we should consider this to be optimal option.



from statsmodels.tsa.statespace.sarimax import SARIMAX

mod = sm.tsa.statespace.SARIMAX(monthly_avg_sales,

                                order=(1, 1, 1),

                                seasonal_order=(0, 1, 1, 12),

                                enforce_stationarity=False,

                                enforce_invertibility=False)

results = mod.fit()

print(results.summary().tables[1])
#  Analysis of Co-efficiecnt and Standrad Error by interpeting the above Result.

#coeff: Shows weight/impotance how each feature impacts the time series. 

#Pvalue: Shows the significance of each feature weight. Can test hypothesis using this. If p value is <.05 then they are statitically significant.



#Refresher on null hyp and pvalues. By default we take the null hyp as 'there is no relationship bw them' If p value < .05 (significance level) then you reject the Null Hypthesis If p value > .05 , then you fail to reject the Null Hypothesis.



#So, if the p-value is < .05 then there is a relationship between the response and predictor. Hence, significant.



#Plotting the diagnostics.



#The plot_diagnostics object allows us to quickly generate model diagnostics and investigate for any unusual behavior.

results.plot_diagnostics(figsize=(20, 10))

plt.show()



#What are the details for analysis and check?

#1. Residuals SHOULD be Normally Distributed ; Check

#Top Right: The (orange colored) KDE line should be closely matched with green colored N(0,1) line. This is the standard notation

#for normal distribution with mean 0 and sd 1.

#Bottom Left: The qq plot shows the ordered distribution of residuals (blue dots) follows the linear trend of the samples 

#taken from a standard normal distribution with N(0, 1). 



#2. #Residuals are not correlated; Check

#Top Left: The standard residuals don’t display any obvious seasonality and appear to be white noise. 

#Bottom Right: The autocorrelation (i.e. correlogram) plot on the bottom right, which shows that the time series residuals have 

#low correlation with its own lagged versions.
#MODEL Evaluation and Analysis

#Lets get the predictions and confidence interval for those predictions.

#Get the predictions. The forecasts start from the 1st of Jan 2017 but the previous line shows how it fits to the data.

pred = results.get_prediction(start=pd.to_datetime('2014-05-01'), dynamic=False) #false is when using the entire history.

#Confidence interval.

pred_ci = pred.conf_int()



#Plotting real and forecasted values.

ax = monthly_avg_sales['2016':].plot(label='observed')

pred.predicted_mean.plot(ax=ax, label='One-step ahead Forecast', alpha=.7, figsize=(14, 7))

ax.fill_between(pred_ci.index,

                pred_ci.iloc[:, 0],

                pred_ci.iloc[:, 1], color='blue', alpha=.2)

ax.set_xlabel('Date')

ax.set_ylabel('Order_Demand')

plt.legend()

plt.show()



#Takeaway: The forecats seems to be fitting well to the data. The Blue/purple thicker plot shows the confidence level in the forecasts. 
#Calculating the Forecast Accuracy



#Calculating the mean squared error (average error of forecasts) and the lower Mean Square Error always reflects the better results 

y_forecasted = pred.predicted_mean

y_truth = monthly_avg_sales['2016-01-01':]

mse = ((y_forecasted - y_truth) ** 2).mean()

print('MSE {}'.format(round(mse, 2)))

print('RMSE: {}'.format(round(np.sqrt(mse), 2)))
#We can make more changes in the time series by using below steps.

pred_uc = results.get_forecast(steps=75)

pred_ci = pred_uc.conf_int()

ax = monthly_avg_sales.plot(label='observed', figsize=(16, 8))

pred_uc.predicted_mean.plot(ax=ax, label='Forecast')

ax.fill_between(pred_ci.index,

                pred_ci.iloc[:, 0],

                pred_ci.iloc[:, 1], color='k', alpha=.25)

ax.set_xlabel('Date')

ax.set_ylabel('Order_Demand')

plt.legend()

plt.show()



#Far out values are naturally more prone to greater variance. 

#The grey area is the confidence we have in the predictions and the corealtes to .