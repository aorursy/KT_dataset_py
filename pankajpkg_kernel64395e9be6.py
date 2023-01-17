import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns

import statsmodels.api as sm

from statsmodels.tsa.stattools import adfuller

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

from sklearn.metrics import mean_squared_error

from math import sqrt

import warnings

warnings.filterwarnings('ignore')

%matplotlib inline

import os

print(os.listdir("../input"))
# Reading and transforming the file

from datetime import datetime

dateparse = lambda x: pd.to_datetime(x, format='%Y%m', errors = 'coerce')

d = pd.read_csv("../input/RainDataNew.csv", parse_dates=['YYYYMM'], index_col='YYYYMM', date_parser=dateparse) 

d.head()

ts = d[pd.Series(pd.to_datetime(d.index, errors='coerce')).notnull().values]

#indexed=d.set_index(['YYYYMM'])

ts.head(15)

rio=ts

#indexed.head()
plt.figure(figsize=(22,6))

sns.lineplot(x=rio.index, y=rio['Data'])

plt.title('Rainfall Variation from 19974 to 2005')

plt.show()
rio['month'] = rio.index.month

rio['year'] = rio.index.year

pivot = pd.pivot_table(rio, values='Data', index='month', columns='year', aggfunc='mean')

pivot.plot(figsize=(20,6))

plt.title('Yearly Rainfall')

plt.xlabel('Months')

plt.ylabel('RainFall')

plt.xticks([x for x in range(1,13)])

plt.legend().remove()

plt.show()
monthly_seasonality = pivot.mean(axis=1)

monthly_seasonality.plot(figsize=(20,6))

plt.title('Monthly Rainfall ')

plt.xlabel('Months')

plt.ylabel('Rainfall')

plt.xticks([x for x in range(1,13)])

plt.show()
year_avg = pd.pivot_table(rio, values='Data', index='year', aggfunc='mean')

year_avg['2 Years MA'] = year_avg['Data'].rolling(2).mean()

year_avg[['Data','2 Years MA']].plot(figsize=(20,6))

plt.title('Yearly AVG Rain')

plt.xlabel('Months')

plt.ylabel('Rainfall')

plt.xticks([x for x in range(1974,2005,1)])

plt.show()
#NO Trend

train = rio[:-60].copy()

val = rio[-60:-12].copy()

test = rio[-12:].copy()

val.tail()
baseline = val['Data'].shift(12)

baseline.dropna(inplace=True)

baseline.head()



ilo=val.iloc[1:,0]

ilo.head()
def measure_rmse(y_true, y_pred):

    return sqrt(mean_squared_error(y_true,y_pred))



# Using the function with the baseline values

rmse_base = measure_rmse(val.iloc[12:,0],baseline)

print(f'The RMSE of the baseline that we will try to diminish is {round(rmse_base,4)} in mm')
def check_stationarity(y, lags_plots=48, figsize=(22,8)):

    "Use Series as parameter"

    

    # Creating plots of the DF

    y = pd.Series(y)

    fig = plt.figure()



    ax1 = plt.subplot2grid((3, 3), (0, 0), colspan=2)

    ax2 = plt.subplot2grid((3, 3), (1, 0))

    ax3 = plt.subplot2grid((3, 3), (1, 1))

    ax4 = plt.subplot2grid((3, 3), (2, 0), colspan=2)



    y.plot(ax=ax1, figsize=figsize)

    ax1.set_title('Rainfall Variation')

    plot_acf(y, lags=lags_plots, zero=False, ax=ax2);

    plot_pacf(y, lags=lags_plots, zero=False, ax=ax3);

    sns.distplot(y, bins=int(sqrt(len(y))), ax=ax4)

    ax4.set_title('Distribution Chart')



    plt.tight_layout()

    

    print('Results of Dickey-Fuller Test:')

    adfinput = adfuller(y)

    adftest = pd.Series(adfinput[0:4], index=['Test Statistic','p-value','Lags Used','Number of Observations Used'])

    adftest = round(adftest,4)

    

    for key, value in adfinput[4].items():

        adftest["Critical Value (%s)"%key] = value.round(4)

        

    print(adftest)

    

    if adftest[0].round(2) < adftest[5].round(2):

        print('\nThe Test Statistics is lower than the Critical Value of 5%.\nThe serie seems to be stationary')

    else:

        print("\nThe Test Statistics is higher than the Critical Value of 5%.\nThe serie isn't stationary")
# The first approach is to check the series without any transformation

check_stationarity(train['Data'])
check_stationarity(train['Data'].diff(12).dropna())
def walk_forward(training_set, validation_set, params):

    '''

    Params: it's a tuple where you put together the following SARIMA parameters: ((pdq), (PDQS), trend)

    '''

    history = [x for x in training_set.values]

    prediction = list()

    

    # Using the SARIMA parameters and fitting the data

    pdq, PDQS, trend = params



    #Forecasting one period ahead in the validation set

    for week in range(len(validation_set)):

        model = sm.tsa.statespace.SARIMAX(history, order=pdq, seasonal_order=PDQS, trend=trend)

        result = model.fit(disp=False)

        yhat = result.predict(start=len(history), end=len(history))

        prediction.append(yhat[0])

        history.append(validation_set[week])

        

    return prediction
# Let's test it in the validation set

val['Pred'] = walk_forward(train['Data'], val['Data'], ((4,0,1),(0,1,1,12),'c'))
# Measuring the error of the prediction

rmse_pred = measure_rmse(val['Data'], val['Pred'])



print(f"The RMSE of the SARIMA(3,0,0),(0,1,1,12),'c' model was {round(rmse_pred,4)} in mm")

print(f"It's a decrease of {round((rmse_pred/rmse_base-1)*100,2)}% in the RMSE")
# Creating the error column

val['Error'] = val['Data'] - val['Pred']
def plot_error(data, figsize=(20,8)):

    '''

    There must have 3 columns following this order: Rainfall, Prediction, Error

    '''

    plt.figure(figsize=figsize)

    ax1 = plt.subplot2grid((2,2), (0,0))

    ax2 = plt.subplot2grid((2,2), (0,1))

    ax3 = plt.subplot2grid((2,2), (1,0))

    ax4 = plt.subplot2grid((2,2), (1,1))

    

    #Plotting the Current and Predicted values

    ax1.plot(data.iloc[:,0:2])

    ax1.legend(['Real','Pred'])

    ax1.set_title('Current and Predicted Values')

    

    # Residual vs Predicted values

    ax2.scatter(data.iloc[:,1], data.iloc[:,2])

    ax2.set_xlabel('Predicted Values')

    ax2.set_ylabel('Errors')

    ax2.set_title('Errors versus Predicted Values')

    

    ## QQ Plot of the residual

    sm.graphics.qqplot(data.iloc[:,2], line='r', ax=ax3)

    

    # Autocorrelation plot of the residual

    plot_acf(data.iloc[:,2], lags=(len(data.iloc[:,2])-1),zero=False, ax=ax4)

    plt.tight_layout()

    plt.show()
# We need to remove some columns to plot the charts

val.drop(['month','year'], axis=1, inplace=True)

val.head()
plot_error(val)
#Creating the new concatenating the training and validation set:

future = pd.concat([train['Data'], val['Data']])

future.head()
# Using the same parameters of the fitted model

model = sm.tsa.statespace.SARIMAX(future, order=(4,0,3), seasonal_order=(0,1,1,12), trend='c')

result = model.fit(disp=False)
test['Pred'] = result.predict(start=(len(future)), end=(len(future)+13))
test[['Data', 'Pred']].plot(figsize=(22,6))

plt.title('Current Values compared to the Extrapolated Ones')

plt.show()
test.info()
test_baseline = test['Data'].shift()



test_baseline[0] = test['Data'][0]



rmse_test_base = measure_rmse(test['Data'],test_baseline)

rmse_test_extrap = measure_rmse(test['Data'], test['Pred'])



print(f'The baseline RMSE for the test baseline was {round(rmse_test_base,2)} in mm')

print(f'The baseline RMSE for the test extrapolation was {round(rmse_test_extrap,2)} in mm')

print(f'That is an improvement of {-round((rmse_test_extrap/rmse_test_base-1)*100,2)}%')