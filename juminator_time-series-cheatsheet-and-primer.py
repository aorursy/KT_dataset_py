import numpy as np                               

import pandas as pd                              

import matplotlib.pyplot as plt                  

import seaborn as sns                            

from dateutil.relativedelta import relativedelta 

from scipy.optimize import minimize              

import statsmodels.formula.api as smf            

import statsmodels.tsa.api as smt

import statsmodels.api as sm

import scipy.stats as scs

from itertools import product                    

from tqdm import tqdm_notebook



from statsmodels.tsa.arima_model import ARMA

from statsmodels.tsa.stattools import adfuller

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

from statsmodels.tsa.arima_process import ArmaProcess

from statsmodels.tsa.arima_model import ARIMA

import math



%matplotlib inline
## Evaluation Metrics



from sklearn.metrics import r2_score, median_absolute_error, mean_absolute_error

from sklearn.metrics import median_absolute_error, mean_squared_error, mean_squared_log_error



def mean_absolute_percentage_error(y_true, y_pred): 

    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
data = pd.read_csv("../input/sandp500/all_stocks_5yr.csv")
df = data.copy()[data.Name=='AAPL'][['date','close']]
df['date_dt'] = pd.to_datetime(df.date)
df.info()
# set the datetime column as the index

df.set_index('date_dt', inplace=True)
# datetimeindex with daily frequency

dti_df = pd.date_range(start='7/1/20', end='7/8/20')



dti_df
# Creating a datetimeindex with monthly frequency

dti_mf = pd.date_range(start='1/1/20', end='7/1/20', freq='M')



dti_mf
df["close"].asfreq('D').plot(legend=True)

lagged_plot = df["close"].asfreq('D').shift(90).plot(legend=True)

lagged_plot.legend(['Apple Inc.','Apple Inc. Lagged'])
# Downsampling (from daily to monthly; aggregation method: mean)



df['close'].resample('M').mean()
# Upsampling (from monthly mean to daily frequency: In this case, filling daily values of the same month with the mean value for that month)



monthly = df['close'].resample('M').mean()

monthly.resample('D').pad()
# Percent Change from previous value (Stock price today / stock price yesterday)

df['ch'] = df.close.div(df.close.shift())

df['ch'].plot(figsize=(13,5))
# Returns

df.close.pct_change().mul(100).plot(figsize=(15,6))
# Absolute change in price

df.close.diff().plot(figsize=(12,5))
# Using the rolling( ) function

rolling = df.close.rolling('200D').mean()

df.close.plot()

rolling.plot()

plt.legend(['Close Price','Rolling Close Price Average'])
# using the expanding( ) function



expanding_mean = df.close.expanding(90).mean() # average of 90 previous values and itself

expanding_std = df.close.expanding(90).std() # std of 90 previous values and itself



df.close.plot()

expanding_mean.plot()

expanding_std.plot()

plt.legend(['Close Price','Expanding Mean','Expanding Standard Deviation'])
# Calculate average of last n obervations Using self-made function



def moving_average(series, n):

    """

        Calculate average of last n observations

    """

    return np.average(series[-n:])



moving_average(df.close, 90)
# Filter only Mondays

mondays = df[df.index.dayofweek.isin([0])]
mondays.head(10)
df['index'] = df.index
df["Year"] = df["index"].dt.year

df["Month"] = df["index"].dt.month

df["Day"] = df["index"].dt.day

df["Hour"] = df["index"].dt.hour

df["Minute"] = df["index"].dt.minute

df["Second"] = df["index"].dt.second

df["Nanosecond"] = df["index"].dt.nanosecond

df["Date"] = df["index"].dt.date

df["Time"] = df["index"].dt.time

df["Time_Time_Zone"] = df["index"].dt.timetz

df["Day_Of_Year"] = df["index"].dt.dayofyear

df["Week_Of_Year"] = df["index"].dt.weekofyear

df["Week"] = df["index"].dt.week

df["Day_Of_week"] = df["index"].dt.dayofweek

df["Week_Day"] = df["index"].dt.weekday

# df["Week_Day_Name"] = df["index"].dt.weekday_name

df["Quarter"] = df["index"].dt.quarter

df["Days_In_Month"] = df["index"].dt.days_in_month

df["Is_Month_Start"] = df["index"].dt.is_month_start

df["Is_Month_End"] = df["index"].dt.is_month_end

df["Is_Quarter_Start"] = df["index"].dt.is_quarter_start

df["Is_Quarter_End"] = df["index"].dt.is_quarter_end

df["Is_Leap_Year"] = df["index"].dt.is_leap_year
def plotMovingAverage(series, window, plot_intervals=False, scale=1.96, plot_anomalies=False):



    """

        series - dataframe with timeseries

        window - rolling window size 

        plot_intervals - show confidence intervals

        plot_anomalies - show anomalies 



    """

    rolling_mean = series.rolling(window=window).mean()



    plt.figure(figsize=(15,5))

    plt.title("Moving average\n window size = {}".format(window))

    plt.plot(rolling_mean, "g", label="Rolling mean trend")



    # Plot confidence intervals for smoothed values

    if plot_intervals:

        mae = mean_absolute_error(series[window:], rolling_mean[window:])

        deviation = np.std(series[window:] - rolling_mean[window:])

        lower_bond = rolling_mean - (mae + scale * deviation)

        upper_bond = rolling_mean + (mae + scale * deviation)

        plt.plot(upper_bond, "r--", label="Upper Bond / Lower Bond")

        plt.plot(lower_bond, "r--")

        

        # Having the intervals, find abnormal values

        if plot_anomalies:

            anomalies = pd.DataFrame(index=series.index, columns=series.columns)

            anomalies[series<lower_bond] = series[series<lower_bond]

            anomalies[series>upper_bond] = series[series>upper_bond]

            plt.plot(anomalies, "ro", markersize=10)

        

    plt.plot(series[window:], label="Actual values")

    plt.legend(loc="upper left")

    plt.grid(True)
plotMovingAverage(df[['close']], 90, plot_intervals=True, scale=1.96, plot_anomalies=True)
data['date_dt'] = pd.to_datetime(data.date)

data.set_index('date_dt',inplace=True)
data_appl = data.copy()[data.Name == 'AAPL']
plt.style.use('fivethirtyeight') 

%matplotlib inline

from pylab import rcParams

from plotly import tools

from plotly.offline import init_notebook_mode, iplot

init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.figure_factory as ff



# OHLC chart of Feb 2018

trace = go.Ohlc(x=data_appl['2013-02'].index,

                open=data_appl['2013-02'].open,

                high=data_appl['2013-02'].high,

                low=data_appl['2013-02'].low,

                close=data_appl['2013-02'].close)

dataa = [trace]

iplot(dataa, filename='simple_ohlc')
# Candlestick chart of Feb 2018

trace = go.Candlestick(x=data_appl['2013-02'].index,

                open=data_appl['2013-02'].open,

                high=data_appl['2013-02'].high,

                low=data_appl['2013-02'].low,

                close=data_appl['2013-02'].close)

dataa = [trace]

iplot(dataa, filename='simple_candlestick')
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
plot_acf(df["close"], lags=50, title="AutoCorrelation Plot")
# using pandas plotting lib

pd.plotting.autocorrelation_plot(df.close)
plot_pacf(df["close"],lags=50)
# The original non-stationary plot

df.close.plot(figsize=(13,6))
sm.tsa.seasonal_decompose(df.close, period=365).plot()

print("Dickey-Fuller criterion: p=%f" % 

      sm.tsa.stattools.adfuller(df.close)[1])
sm.tsa.seasonal_decompose(df['diff1'].dropna(), period=365).plot()



print("Dickey-Fuller criterion: p=%f" % 

      sm.tsa.stattools.adfuller(df['diff1'].dropna())[1])