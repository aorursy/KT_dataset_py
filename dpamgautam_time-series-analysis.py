import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

print(os.listdir("../input"))
import matplotlib.pyplot as plt

import seaborn as sns



from dateutil.relativedelta import relativedelta

from scipy.optimize import minimize



import statsmodels.formula.api as smf            # statistics and econometrics

import statsmodels.tsa.api as smt

import statsmodels.api as sm

import scipy.stats as scs



from itertools import product                    # some useful functions

from tqdm import tqdm_notebook



import warnings                                  # `do not disturbe` mode

warnings.filterwarnings('ignore')



%matplotlib inline
ads = pd.read_csv("../input/ads.csv", index_col=['Time'], parse_dates=['Time'])

currency = pd.read_csv("../input/currency.csv", index_col=['Time'], parse_dates=['Time'])
ads.head(5)
currency.head(5)
plt.figure(figsize=(12,5))

plt.plot(ads.Ads)

plt.title('Ads watched (hourly data)')

plt.grid(True)

plt.show()



plt.figure(figsize=(12,5))

plt.plot(currency.GEMS_GEMS_SPENT)

plt.title("In game currency spent (daily data)")

plt.grid(True)

plt.show()
from sklearn.metrics import r2_score, mean_absolute_error, median_absolute_error

from sklearn.metrics import mean_squared_error, mean_squared_log_error



def mean_absolute_percentage_error(y_true, y_pred):

    return np.mean(np.abs((y_true - y_pred) / y_true))*100
def moving_average(series, n):

    return np.average(series[-n:])



moving_average(ads, 24)
def plotMovingAverage(series, window, plot_intervals=False, scale=1.96, plot_anomalies=False):

    

    rolling_mean = series.rolling(window=window).mean()

    

    plt.figure(figsize=(12,5))

    plt.title("Moving Average\nwindow size = {}".format(window))

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

    plt.legend(loc='upper left')

    plt.grid(True)
plotMovingAverage(ads, 4)
plotMovingAverage(ads, 12)
plotMovingAverage(ads, 24)
plotMovingAverage(ads, 4, plot_intervals=True)
ads_anomaly = ads.copy()

ads_anomaly.iloc[-20] = ads_anomaly.iloc[-20]*0.2
plotMovingAverage(ads_anomaly, 4, plot_intervals=True, plot_anomalies=True)
plotMovingAverage(currency, 7, plot_intervals=True, plot_anomalies=True)
def weighted_average(series, weights):

    """

        Calculate weighter average on series

    """

    result = 0.0

    weights.reverse()

    for n in range(len(weights)):

        result += series.iloc[-n-1] * weights[n]

    return float(result)

  

weighted_average(ads, [0.6, 0.3, 0.1])
def exponential_smoothing(series, alpha):

    """

        series - dataset with timestamps

        alpha - float [0.0, 1.0], smoothing parameter

    """

    result = [series[0]] # first value is same as series

    for n in range(1, len(series)):

        result.append(alpha * series[n] + (1 - alpha) * result[n-1])

    return result



def plotExponentialSmoothing(series, alphas):

    """

        Plots exponential smoothing with different alphas

        

        series - dataset with timestamps

        alphas - list of floats, smoothing parameters

        

    """

    with plt.style.context('seaborn-white'):    

        plt.figure(figsize=(12,5))

        for alpha in alphas:

            plt.plot(exponential_smoothing(series, alpha), label="Alpha {}".format(alpha))

        plt.plot(series.values, "c", label = "Actual")

        plt.legend(loc="best")

        plt.axis('tight')

        plt.title("Exponential Smoothing")

        plt.grid(True);
plotExponentialSmoothing(ads.Ads, [0.3, 0.05])

plotExponentialSmoothing(currency.GEMS_GEMS_SPENT, [0.3, 0.05])