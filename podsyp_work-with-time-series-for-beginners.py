# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import matplotlib.pyplot as plt

import seaborn as sns

sns.set()



from dateutil.relativedelta import relativedelta

from scipy.optimize import minimize



import statsmodels.formula.api as smf

import statsmodels.tsa.api as smt

import statsmodels.api as sm

import scipy.stats as scs



from itertools import product

from tqdm import tqdm_notebook



import warnings

warnings.filterwarnings('ignore')



%matplotlib inline
from sklearn.metrics import r2_score, median_absolute_error, mean_absolute_error

from sklearn.metrics import median_absolute_error, mean_squared_error, mean_squared_log_error



def mean_absolute_percentage_error(y_true, y_pred): 

    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
full_df = pd.read_csv('/kaggle/input/time-series-starter-dataset/Month_Value_1.csv', sep=',')
full_df.head()
df = full_df[['Revenue']].dropna()
plt.figure(figsize=(18, 6))

plt.plot(df.Revenue)

plt.title('Revenue (month data)')

plt.grid(True)

plt.show()
def moving_average(series, n):

    """

        Calculate average of last n observations

    """

    return np.average(series[-n:])



moving_average(df, 12) # prediction for the last observed day (past 24 hours)
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
plotMovingAverage(df, 4) 
plotMovingAverage(df, 12) 
plotMovingAverage(df, 4, plot_intervals=True)
plotMovingAverage(df, 12, plot_intervals=True)
def weighted_average(series, weights):

    """

        Calculate weighter average on series

    """

    result = 0.0

    weights.reverse()

    for n in range(len(weights)):

        result += series.iloc[-n-1] * weights[n]

    return float(result)
weighted_average(df, [0.6, 0.3, 0.1])
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

        plt.figure(figsize=(15, 7))

        for alpha in alphas:

            plt.plot(exponential_smoothing(series, alpha), label="Alpha {}".format(alpha))

        plt.plot(series.values, "c", label = "Actual")

        plt.legend(loc="best")

        plt.axis('tight')

        plt.title("Exponential Smoothing")

        plt.grid(True);
plotExponentialSmoothing(df.Revenue, [0.3, 0.05])
def double_exponential_smoothing(series, alpha, beta):

    """

        series - dataset with timeseries

        alpha - float [0.0, 1.0], smoothing parameter for level

        beta - float [0.0, 1.0], smoothing parameter for trend

    """

    # first value is same as series

    result = [series[0]]

    for n in range(1, len(series)+1):

        if n == 1:

            level, trend = series[0], series[1] - series[0]

        if n >= len(series): # forecasting

            value = result[-1]

        else:

            value = series[n]

        last_level, level = level, alpha*value + (1-alpha)*(level+trend)

        trend = beta*(level-last_level) + (1-beta)*trend

        result.append(level+trend)

    return result



def plotDoubleExponentialSmoothing(series, alphas, betas):

    """

        Plots double exponential smoothing with different alphas and betas

        

        series - dataset with timestamps

        alphas - list of floats, smoothing parameters for level

        betas - list of floats, smoothing parameters for trend

    """

    

    with plt.style.context('seaborn-white'):    

        plt.figure(figsize=(20, 8))

        for alpha in alphas:

            for beta in betas:

                plt.plot(double_exponential_smoothing(series, alpha, beta), label="Alpha {}, beta {}".format(alpha, beta))

        plt.plot(series.values, label = "Actual")

        plt.legend(loc="best")

        plt.axis('tight')

        plt.title("Double Exponential Smoothing")

        plt.grid(True)
plotDoubleExponentialSmoothing(df.Revenue, alphas=[0.9, 0.02], betas=[0.9, 0.02])
from sklearn.model_selection import TimeSeriesSplit # you have everything done for you



def timeseriesCVscore(params, series, loss_function=mean_squared_error, slen=24):

    """

        Returns error on CV  

        

        params - vector of parameters for optimization

        series - dataset with timeseries

        slen - season length for Holt-Winters model

    """

    # errors array

    errors = []

    

    values = series.values

    alpha, beta, gamma = params

    

    # set the number of folds for cross-validation

    tscv = TimeSeriesSplit(n_splits=3) 

    

    # iterating over folds, train model on each, forecast and calculate error

    for train, test in tscv.split(values):



        model = HoltWinters(series=values[train], slen=slen, 

                            alpha=alpha, beta=beta, gamma=gamma, n_preds=len(test))

        model.triple_exponential_smoothing()

        

        predictions = model.result[-len(test):]

        actual = values[test]

        error = loss_function(predictions, actual)

        errors.append(error)

        

    return np.mean(np.array(errors))
class HoltWinters:

    

    """

    Holt-Winters model with the anomalies detection using Brutlag method

    

    # series - initial time series

    # slen - length of a season

    # alpha, beta, gamma - Holt-Winters model coefficients

    # n_preds - predictions horizon

    # scaling_factor - sets the width of the confidence interval by Brutlag (usually takes values from 2 to 3)

    

    """

    

    

    def __init__(self, series, slen, alpha, beta, gamma, n_preds, scaling_factor=1.96):

        self.series = series

        self.slen = slen

        self.alpha = alpha

        self.beta = beta

        self.gamma = gamma

        self.n_preds = n_preds

        self.scaling_factor = scaling_factor

        

        

    def initial_trend(self):

        sum = 0.0

        for i in range(self.slen):

            sum += float(self.series[i+self.slen] - self.series[i]) / self.slen

        return sum / self.slen  

    

    def initial_seasonal_components(self):

        seasonals = {}

        season_averages = []

        n_seasons = int(len(self.series)/self.slen)

        # let's calculate season averages

        for j in range(n_seasons):

            season_averages.append(sum(self.series[self.slen*j:self.slen*j+self.slen])/float(self.slen))

        # let's calculate initial values

        for i in range(self.slen):

            sum_of_vals_over_avg = 0.0

            for j in range(n_seasons):

                sum_of_vals_over_avg += self.series[self.slen*j+i]-season_averages[j]

            seasonals[i] = sum_of_vals_over_avg/n_seasons

        return seasonals   



          

    def triple_exponential_smoothing(self):

        self.result = []

        self.Smooth = []

        self.Season = []

        self.Trend = []

        self.PredictedDeviation = []

        self.UpperBond = []

        self.LowerBond = []

        

        seasonals = self.initial_seasonal_components()

        

        for i in range(len(self.series)+self.n_preds):

            if i == 0: # components initialization

                smooth = self.series[0]

                trend = self.initial_trend()

                self.result.append(self.series[0])

                self.Smooth.append(smooth)

                self.Trend.append(trend)

                self.Season.append(seasonals[i%self.slen])

                

                self.PredictedDeviation.append(0)

                

                self.UpperBond.append(self.result[0] + 

                                      self.scaling_factor * 

                                      self.PredictedDeviation[0])

                

                self.LowerBond.append(self.result[0] - 

                                      self.scaling_factor * 

                                      self.PredictedDeviation[0])

                continue

                

            if i >= len(self.series): # predicting

                m = i - len(self.series) + 1

                self.result.append((smooth + m*trend) + seasonals[i%self.slen])

                

                # when predicting we increase uncertainty on each step

                self.PredictedDeviation.append(self.PredictedDeviation[-1]*1.01) 

                

            else:

                val = self.series[i]

                last_smooth, smooth = smooth, self.alpha*(val-seasonals[i%self.slen]) + (1-self.alpha)*(smooth+trend)

                trend = self.beta * (smooth-last_smooth) + (1-self.beta)*trend

                seasonals[i%self.slen] = self.gamma*(val-smooth) + (1-self.gamma)*seasonals[i%self.slen]

                self.result.append(smooth+trend+seasonals[i%self.slen])

                

                # Deviation is calculated according to Brutlag algorithm.

                self.PredictedDeviation.append(self.gamma * np.abs(self.series[i] - self.result[i]) 

                                               + (1-self.gamma)*self.PredictedDeviation[-1])

                     

            self.UpperBond.append(self.result[-1] + 

                                  self.scaling_factor * 

                                  self.PredictedDeviation[-1])



            self.LowerBond.append(self.result[-1] - 

                                  self.scaling_factor * 

                                  self.PredictedDeviation[-1])



            self.Smooth.append(smooth)

            self.Trend.append(trend)

            self.Season.append(seasonals[i%self.slen])
def tsplot(y, lags=None, figsize=(12, 7), style='bmh'):

    """

        Plot time series, its ACF and PACF, calculate Dickey–Fuller test

        

        y - timeseries

        lags - how many lags to include in ACF, PACF calculation

    """

    if not isinstance(y, pd.Series):

        y = pd.Series(y)

        

    with plt.style.context(style):    

        fig = plt.figure(figsize=figsize)

        layout = (2, 2)

        ts_ax = plt.subplot2grid(layout, (0, 0), colspan=2)

        acf_ax = plt.subplot2grid(layout, (1, 0))

        pacf_ax = plt.subplot2grid(layout, (1, 1))

        

        y.plot(ax=ts_ax)

        p_value = sm.tsa.stattools.adfuller(y)[1]

        ts_ax.set_title('Time Series Analysis Plots\n Dickey-Fuller: p={0:.5f}'.format(p_value))

        smt.graphics.plot_acf(y, lags=lags, ax=acf_ax)

        smt.graphics.plot_pacf(y, lags=lags, ax=pacf_ax)

        plt.tight_layout()
tsplot(df.Revenue, lags=12)
df_diff = df.Revenue - df.Revenue.shift(12)

tsplot(df_diff[12:], lags=6)