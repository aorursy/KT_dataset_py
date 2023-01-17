import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

import gc
#importing all input data to Dataframe
df_holevents = pd.read_csv('input_data/holidays_events.csv')

df_items     = pd.read_csv('input_data/items.csv')

df_oil       = pd.read_csv('input_data/oil.csv')

df_stores    = pd.read_csv('input_data/stores.csv')

df_test      = pd.read_csv('input_data/test.csv')

df_train     = pd.read_csv('input_data/train.csv')

df_trans     = pd.read_csv('input_data/transactions.csv')
print(df_train.shape)

df_train.head()
print(df_test.shape)

df_test.head()
print(df_stores.shape)

df_stores.head()
print(df_items.shape)

df_items.head()
print(df_oil.shape)

df_oil.head()
print(df_trans.shape)

df_trans.head()
df_train.info()
#Now, let's get a sense of the time range for which the data was collected.
# convert date to datetime

df_train["date"] =  pd.to_datetime(df_train["date"])
df_train.head()
df_train["date"].dt.year.value_counts(sort = False).plot.bar()

plt.xlabel('Year')

plt.ylabel("Collected Data")

df_train_2016 = df_train[df_train["date"].dt.year == 2016]
df_train_2016["date"].dt.month.value_counts(sort = False).plot.bar()
df_train_2016["date"].dt.day.value_counts(sort = False).plot.bar()
# How many stores?
df_stores.head()
df_stores['store_nbr'].unique()
df_stores['state'].unique()
df_train_2016["item_nbr"].unique().shape[0]
stores = np.arange(1, 55)

items_store = np.zeros((54, ))

for i, store in enumerate(stores) :

    items_store[i] = df_train_2016["item_nbr"][df_train_2016["store_nbr"] \

                                               == store].unique().shape[0]

sns.barplot(stores, items_store)
#Item sales is our target variable

df_train_2016["unit_sales"].describe()
df_train_2016["onpromotion"].value_counts()
3514584/31715287 * 100
df_train_2016.isnull().sum()
unit_sales = df_train_2016["unit_sales"].values

gc.collect()
plt.scatter(x = range(unit_sales.shape[0]), y = np.sort(unit_sales))
df_train = df_train.set_index('date')
df_train = df_train[['unit_sales']]
df_train = df_train.to_period(freq='M')

df_train.head()
df_train = df_train.groupby(['date']).sum()
df_train.head()
#Plotting the time Series for the training dataset

df_train.plot()
df_train.plot(kind = "hist", bins = 30)
df_train['sales_unit_Log'] = np.log(df_train.unit_sales)

df_train.head()
df_train['sales_unit_Log'].plot(kind = "hist", bins = 30)
df_train['sales_unit_Log'].plot()
model_mean_pred = df_train.sales_unit_Log.mean()
# Let us store this as our Mean Predication Value

df_train["salesMean"] = np.exp(model_mean_pred)
df_train.head()
df_train.plot(kind="line", y = ["unit_sales", "salesMean"])
def RMSE(predicted, actual):

    mse = (predicted - actual)**2

    rmse = np.sqrt(mse.sum()/mse.count())

    return rmse
model_mean_RMSE = RMSE(df_train.salesMean, df_train.unit_sales)

model_mean_RMSE
# Save this in a dataframe

dfBangResults = pd.DataFrame(columns = ["Model", "Forecast", "RMSE"])

dfBangResults.head()
dfBangResults.loc[0,"Model"] = "Mean"

dfBangResults.loc[0,"Forecast"] = np.exp(model_mean_pred)

dfBangResults.loc[0,"RMSE"] = model_mean_RMSE

dfBangResults.head()
df_train.head()
df_train['date'] = df_train.index.to_timestamp()
df_train.head()
# Convert date in datetimedelta figure starting from zero

df_train["timeIndex"] = df_train.date - df_train.date.min()
df_train.head()
df_train.dtypes
# Convert to months using the timedelta function

df_train["timeIndex"] =  df_train["timeIndex"]/np.timedelta64(1, 'M')
df_train.timeIndex.head()
# Round the number to 0

df_train["timeIndex"] = df_train["timeIndex"].round(0).astype(int)
df_train.tail()
## Now plot linear regression

# Import statsmodel

import statsmodels.api as sm

import statsmodels.formula.api as smf

from statsmodels.tsa.stattools import adfuller



model_linear = smf.ols('sales_unit_Log ~ timeIndex', data = df_train).fit()
model_linear.summary()
## Parameters for y = mx + c equation

model_linear.params
c = model_linear.params[0]

c
m = model_linear.params[1]

m
model_linear_pred = model_linear.predict()
model_linear_pred
# Plot the prediction line

df_train.plot(kind="line", x="timeIndex", y = "sales_unit_Log")

plt.plot(df_train.timeIndex,model_linear_pred, '-')
model_linear.resid.plot(kind = "bar")
df_train["salesLinear"] = np.exp(model_linear_pred)
df_train.head()
# Root Mean Squared Error (RMSE)

model_linear_RMSE = RMSE(df_train.salesLinear, df_train.unit_sales)

model_linear_RMSE
# Manual Calculation

model_linear_forecast_manual = m * 146 + c

model_linear_forecast_manual
dfBangResults.loc[1,"Model"] = "Linear"

dfBangResults.loc[1,"Forecast"] = np.exp(model_linear_forecast_manual)

dfBangResults.loc[1,"RMSE"] = model_linear_RMSE

dfBangResults.head()
df_train.plot(kind="line", x="timeIndex", y = ["unit_sales", "salesMean", "salesLinear"])
df_train["priceModLogShift1"] = df_train.sales_unit_Log.shift()
df_train.head()
df_train.plot(kind= "scatter", y = "sales_unit_Log", x = "priceModLogShift1", s = 50)
# Lets plot the one-month difference curve

df_train["priceModLogDiff"] = df_train.sales_unit_Log - df_train.priceModLogShift1
df_train.priceModLogDiff.plot()
df_train["priceRandom"] = np.exp(df_train.priceModLogShift1)

df_train.head()
df_train.plot(kind="line", x="timeIndex", y = ["unit_sales","priceRandom"])
# Root Mean Squared Error (RMSE)

model_random_RMSE = RMSE(df_train.priceRandom, df_train.unit_sales)

model_random_RMSE
dfBangResults.loc[2,"Model"] = "Random"

dfBangResults.loc[2,"Forecast"] = np.exp(df_train.priceModLogShift1[-1])

dfBangResults.loc[2,"RMSE"] = model_random_RMSE

dfBangResults.head()
def adf(ts):

    

    # Determing rolling statistics

    rolmean = pd.rolling_mean(ts, window=12)

    rolstd = pd.rolling_std(ts, window=12)



    #Plot rolling statistics:

    orig = plt.plot(ts, color='blue',label='Original')

    mean = plt.plot(rolmean, color='red', label='Rolling Mean')

    std = plt.plot(rolstd, color='black', label = 'Rolling Std')

    plt.legend(loc='best')

    plt.title('Rolling Mean & Standard Deviation')

    plt.show(block=False)

    

    # Calculate ADF factors

    adftest = adfuller(ts, autolag='AIC')

    adfoutput = pd.Series(adftest[0:4], index=['Test Statistic','p-value','# of Lags Used',

                                              'Number of Observations Used'])

    for key,value in adftest[4].items():

        adfoutput['Critical Value (%s)'%key] = value

    return adfoutput
# For smoothing the values we can use 12 month Moving Averages 

df_train['priceModLogMA12'] = pd.rolling_mean(df_train.sales_unit_Log, window = 12)
df_train.plot(kind ="line", y=["priceModLogMA12", "sales_unit_Log"])
df_train["priceMA12"] = np.exp(df_train.priceModLogMA12)

df_train.tail()
model_MA12_forecast = df_train.sales_unit_Log.tail(12).mean()
# Root Mean Squared Error (RMSE)

model_MA12_RMSE = RMSE(df_train.priceMA12, df_train.unit_sales)

model_MA12_RMSE
dfBangResults.loc[3,"Model"] = "Moving Average 12"

dfBangResults.loc[3,"Forecast"] = np.exp(model_MA12_forecast)

dfBangResults.loc[3,"RMSE"] = model_MA12_RMSE

dfBangResults.head()
df_train.plot(kind="line", x="timeIndex", y = ["unit_sales", "salesMean", "salesLinear",

                                             "priceRandom", "priceMA12"])
df_train.priceModLogDiff.plot()
# Test remaining part for Stationary

ts = df_train.priceModLogDiff

ts.dropna(inplace = True)

adfuller(ts)
from statsmodels.tsa.seasonal import seasonal_decompose

df_train.index = df_train.index.to_datetime()

df_train.head()
decomposition = seasonal_decompose(df_train.sales_unit_Log, model = "additive")

decomposition.plot()
trend = decomposition.trend

seasonal = decomposition.seasonal

residual = decomposition.resid
df_train["priceDecomp"] = np.exp(trend + seasonal)
# Root Mean Squared Error (RMSE)

model_Decomp_RMSE = RMSE(df_train.priceDecomp, df_train.unit_sales)

model_Decomp_RMSE
df_train.plot(kind="line", x="timeIndex", y = ["unit_sales", "salesMean", "salesLinear", "priceRandom",

                                             "priceMA12",  "priceDecomp"])
df_train.plot(kind="line", x="timeIndex", y = ["unit_sales",

                                              "priceDecomp"])
# Test remaining part for Stationary

ts = decomposition.resid

ts.dropna(inplace = True)

adfuller(ts)
ts = df_train.sales_unit_Log

ts_diff = df_train.priceModLogDiff

ts_diff.dropna(inplace = True)
#ACF and PACF plots:

from statsmodels.tsa.stattools import acf, pacf
lag_acf = acf(ts_diff, nlags=20)

lag_acf
ACF = pd.Series(lag_acf)
ACF.plot(kind = "bar")
lag_pacf = pacf(ts_diff, nlags=20, method='ols')
PACF = pd.Series(lag_pacf)

PACF.plot(kind = "bar")
from statsmodels.tsa.arima_model import ARIMA
ts_diff.head()
# Running the ARIMA Model(1,0,1)

model_AR1MA = ARIMA(ts_diff, order=(1,0,1))
results_ARIMA = model_AR1MA.fit(disp = -1)
results_ARIMA.fittedvalues.head()
ts_diff.plot()

results_ARIMA.fittedvalues.plot()
ts_diff.sum()
predictions_ARIMA_diff = pd.Series(results_ARIMA.fittedvalues, copy=True)

predictions_ARIMA_diff.tail()
predictions_ARIMA_diff.sum()
predictions_ARIMA_diff_cumsum = predictions_ARIMA_diff.cumsum()

predictions_ARIMA_diff_cumsum.tail()
ts.ix[0]
predictions_ARIMA_log = pd.Series(ts.ix[0], index=ts.index)

predictions_ARIMA_log = predictions_ARIMA_log.add(predictions_ARIMA_diff_cumsum,fill_value=0)

predictions_ARIMA_log.tail()
df_train['priceARIMA'] = np.exp(predictions_ARIMA_log)
df_train.plot(kind="line", x="timeIndex", y = ["unit_sales", "priceARIMA"])