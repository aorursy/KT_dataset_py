import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import missingno as msno



import datetime

from pandas.tseries.offsets import DateOffset

from pandas.tseries.offsets import MonthEnd



import statsmodels.api as sm

from scipy import stats

import itertools



import gc

import warnings

warnings.filterwarnings("ignore")



plt.style.use("fivethirtyeight")
path = "../input/bitcoin-historical-data/bitstampUSD_1-min_data_2012-01-01_to_2020-04-22.csv"

df = pd.read_csv(path)
df.head()
#Date conversion

df["Timestamp"] = pd.to_datetime(df["Timestamp"], unit="s",origin="unix")
ax, fig = plt.subplots(figsize=(10,5))



msno.bar(df)



ax.text(0.07,1, s="Missing data check", fontsize=32, weight="bold", alpha=0.75)
df.set_index(df["Timestamp"],drop=True,inplace=True)

#Hour

df_hour = df.resample("h").mean()
ax, fig = plt.subplots(figsize = (10,5))



plt.plot(df_hour["Open"], label="Opening price")

plt.plot(df_hour["Close"], label="Closing price")



plt.xticks(alpha=0.75, weight="bold")

plt.yticks(alpha=0.75, weight="bold")



plt.xlabel("Date",alpha=0.75, weight="bold")

plt.ylabel("Price",alpha=0.75, weight="bold")



plt.legend()



plt.text(x=datetime.date(2011, 6, 30), y=22000, s="Hourly opening and closing price of Bitcoin (2012-2020)",

fontsize=15, weight="bold", alpha=0.75)

plt.text(x=datetime.date(2011, 6, 30), y=21000, s="There is no major difference between the mean opening and closing prices.",fontsize=12, alpha=0.75)
#Data



df_hour["hourly_diff"] = df_hour["Close"] - df_hour["Open"]



#Plot

ax, fig = plt.subplots(figsize = (10,5))



plt.plot(df_hour["hourly_diff"])



plt.xticks(alpha=0.75, weight="bold")

plt.yticks(alpha=0.75, weight="bold")



plt.xlabel("Date",alpha=0.75, weight="bold")

plt.ylabel("Price",alpha=0.75, weight="bold")



plt.legend()



plt.text(x=datetime.date(2011, 6, 30), y=25, s="Hourly difference between the opening and closing Bitcoin prices (2012-2020)",

fontsize=15, weight="bold", alpha=0.75)

plt.text(x=datetime.date(2011, 6, 30), y=22, s="Larger price fluctuations started happening in 2018 when Bitcoin started gaining mainstream appeal.",fontsize=12, alpha=0.75)
ax, fig = plt.subplots(figsize = (10,5))



plt.plot(df_hour["Weighted_Price"])



plt.xticks(alpha=0.75, weight="bold")

plt.yticks(alpha=0.75, weight="bold")



plt.xlabel("Date",alpha=0.75, weight="bold")

plt.ylabel("Price",alpha=0.75, weight="bold")



plt.legend()



plt.text(x=datetime.date(2011, 6, 30), y=22000, s="Weighted Price for Bitcoins (2012-2020)",

fontsize=15, weight="bold", alpha=0.75)

plt.text(x=datetime.date(2011, 6, 30), y=21000, s="This is the main metric that we would like to predict.",fontsize=12, alpha=0.75)
#Seasonal Decompose

ax, fig = plt.subplots(figsize=(15,8), sharex=True)



df_month = df.resample("M").mean()

dec = sm.tsa.seasonal_decompose(df_month["Weighted_Price"])





plt.subplot(411)

plt.plot(df_hour["Weighted_Price"], label="Weighted Price")

plt.title("Observed",loc="left", alpha=0.75, fontsize=18)



plt.subplot(412)

plt.plot(dec.trend, label="Trend")

plt.title("Trend",loc="left", alpha=0.75, fontsize=18)



plt.subplot(413)

plt.plot(dec.seasonal, label="Seasonal")

plt.title("Seasonal",loc="left", alpha=0.75, fontsize=18)



plt.subplot(414)

plt.plot(dec.resid, label="Residual")

plt.title("Residual",loc="left", alpha=0.75, fontsize=18)

plt.tight_layout()



plt.text(x=datetime.date(2011, 6, 30), y=63000, s="Seasonal time series decomposition",fontsize=24, weight="bold", alpha=0.75)

plt.text(x=datetime.date(2011, 6, 30), y=60700, s="Decomposition of the weighted price data ranging from 2012 to 2020.",fontsize=18, alpha=0.75)



gc.collect()
print("Dicky-Fuller stationarity test - p: %f" % sm.tsa.adfuller(df_month["Weighted_Price"])[1])
#Box-Cox



df_month["Box-Cox"], _ = stats.boxcox(df_month["Weighted_Price"])

print("Dicky-Fuller stationarity test - p: %f" % sm.tsa.adfuller(df_month["Box-Cox"])[1])
#Automatic Differencing



first_diff = df_month["Weighted_Price"].diff()

print("Dicky-Fuller stationarity test - p: %f" % sm.tsa.adfuller(first_diff[1:])[1])

print("This series is stationary")





df_month["Auto_Diff"] = first_diff
#Data

seasonal_dec = sm.tsa.seasonal_decompose(df_month["Auto_Diff"][1:])



#Seasonal Decompose on stationary series

ax, fig = plt.subplots(figsize=(15,8), sharex=True)



df_month = df.resample("M").mean()

dec = sm.tsa.seasonal_decompose(df_month["Weighted_Price"])





plt.subplot(411)

plt.plot(df_hour["Weighted_Price"], label="Weighted Price")

plt.title("Observed",loc="left", alpha=0.75, fontsize=18)



plt.subplot(412)

plt.plot(seasonal_dec.trend, label="Trend")

plt.title("Trend",loc="left", alpha=0.75, fontsize=18)



plt.subplot(413)

plt.plot(seasonal_dec.seasonal, label="Seasonal")

plt.title("Seasonal",loc="left", alpha=0.75, fontsize=18)



plt.subplot(414)

plt.plot(seasonal_dec.resid, label="Residual")

plt.title("Residual",loc="left", alpha=0.75, fontsize=18)

plt.tight_layout()



plt.text(x=datetime.date(2011, 6, 30), y=63000, s="Seasonal decomposition on stationary time series",fontsize=24, weight="bold", alpha=0.75)

plt.text(x=datetime.date(2011, 6, 30), y=60700, s="Decomposition of the stationary weighted price data ranging from 2012 to 2020.",fontsize=18, alpha=0.75)



gc.collect()
ax, fig = plt.subplots(figsize=(15,10))



plt.subplot(411)

x = sm.graphics.tsa.plot_acf(first_diff[1:], ax=plt.gca())

plt.subplot(412)

y = sm.graphics.tsa.plot_pacf(first_diff[1:],ax=plt.gca())

plt.tight_layout()



gc.collect()

del x,y
###SARIMAX###



#Constructs all possible parameter combinations.

p = d = q = range(0,2)

pdq = list(itertools.product(p,d,q))



seasonal_pdq = [(x[0],x[1],x[2],12) for x in list(itertools.product(p,d,q))]
def sarimax_function(data,pdq,s_pdq):



    """

    The function uses a brute force approach to apply all possible pdq combinations and evaluate the model

    """



    result_list = []

    for param in pdq:

        for s_param in s_pdq:



            model = sm.tsa.statespace.SARIMAX(data, order=param, seasonal_order=s_param,

            enforce_invertibility=False,enforce_stationarity=False)



            results = model.fit()

            result_list.append([param,s_param,results.aic])

            print("ARIMA Parameters: {} x: {}. AIC: {}".format(param,s_param,results.aic))



    return result_list,results
result_list,results = sarimax_function(df_month["Weighted_Price"],pdq,seasonal_pdq)



gc.collect()
#Dataframe of all results and parameters.



results_dataframe = pd.DataFrame(result_list, columns=["dpq","s_dpq","aic"]).sort_values(by="aic")

results_dataframe.head()
model = sm.tsa.statespace.SARIMAX(df_month["Weighted_Price"], order=(0, 1, 1), seasonal_order=(1, 1, 1, 12),

            enforce_invertibility=False,enforce_stationarity=False).fit()

print(model.summary().tables[1])
#Residual analysis

ax, fig = plt.subplots(figsize = (10,5))



model.resid.plot(label="Residual")



plt.xticks(alpha=0.75, weight="bold")

plt.yticks(alpha=0.75, weight="bold")



plt.xlabel("Date",alpha=0.75, weight="bold")

plt.ylabel("Price",alpha=0.75, weight="bold")



plt.legend()



plt.text(x=datetime.date(2011, 6, 30), y=7200, s="Residual Analysis",

fontsize=15, weight="bold", alpha=0.75)

plt.text(x=datetime.date(2011, 6, 30), y=6700, s="Analaysis of the residual values for the best model acording to AIC.",fontsize=12, alpha=0.75)



gc.collect()
x = model.plot_diagnostics(figsize=(18, 8))



gc.collect()

del x
df_month_prediction = df_month[["Weighted_Price"]]



df_month_prediction["Forcasting"] = model.predict(start=pd.to_datetime("2011-12-31"), end=pd.to_datetime("2020-04-30"))
ax, fig = plt.subplots(figsize = (10,5))



plt.plot(df_month_prediction["Forcasting"], ls="--", label="Prediction")

plt.plot(df_month_prediction["Weighted_Price"], label="Actual Data")



plt.xticks(alpha=0.75, weight="bold")

plt.yticks(alpha=0.75, weight="bold")



plt.xlabel("Date",alpha=0.75, weight="bold")

plt.ylabel("Price",alpha=0.75, weight="bold")



plt.legend()



plt.text(x=datetime.date(2011, 6, 30), y=18000, s="Forcasting test of SARIMAX",

fontsize=18, weight="bold", alpha=0.75)

plt.text(x=datetime.date(2011, 6, 30), y=17000, s="Prediction testing of the best SARIMAX model.",fontsize=15, alpha=0.75)



gc.collect()
#Datetimeindex dates I want to predict



future_dates = [df_month_prediction.index[-1] + DateOffset(months = x)for x in range(1,12)]

future_dates = pd.to_datetime(future_dates)  +  MonthEnd(0)

future = pd.DataFrame(index=future_dates)

df_month_prediction = pd.concat([df_month_prediction,future])



gc.collect()
#Prediction



df_month_prediction["Future_forcast"] = model.predict(start=pd.to_datetime("2020-03-31"),end=pd.to_datetime("2021-03-31"))



pred = model.get_prediction(start=pd.to_datetime("2020-03-31"),end=pd.to_datetime("2021-03-31"))

pred_ci = pred.conf_int()



gc.collect()
ax, fig = plt.subplots(figsize=(10,5))



plt.plot(df_month_prediction["Weighted_Price"], label="Actual")

plt.plot(df_month_prediction["Future_forcast"],ls="--", label="Prediction")



plt.fill_between(pred_ci.index,

                pred_ci.iloc[:, 0],

                pred_ci.iloc[:, 1], color='k', alpha=.2)

plt.legend()





plt.xticks(alpha=0.75, weight="bold")

plt.yticks(alpha=0.75, weight="bold")



plt.xlabel("Date",alpha=0.75, weight="bold")

plt.ylabel("Price",alpha=0.75, weight="bold")



plt.legend()



plt.text(x=datetime.date(2011, 6, 30), y=19500, s="SARIMAX Forcasting",

fontsize=18, weight="bold", alpha=0.75)

plt.text(x=datetime.date(2011, 6, 30), y=18500, s="Prediction of the weighted price for the next 12 months.",fontsize=15, alpha=0.75)



gc.collect()