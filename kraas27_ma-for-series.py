from scipy.optimize import minimize

from sklearn.metrics import mean_squared_error

import pandas as pd
import matplotlib.pyplot as plt

from pylab import rcParams

rcParams['figure.figsize'] = (12, 6)
def double_ema_with_preds(series, alpha, beta, n_preds):

    result = [series[0]]

    level, trend = series[0], series[1] - series[0]

    for n in range(1, len(series)):

        value = series[n]

        last_level, level = level, alpha*value + (1-alpha)*(level+trend)

        trend = beta*(level-last_level) + (1-beta)*trend

        result.append(level+trend)

        

    for n in range(n_preds):

        value = result[-1]

        last_level, level = level, alpha*value + (1-alpha)*(level+trend)

        trend = beta*(level-last_level) + (1-beta)*trend

        result.append(level+trend)



    return  result
robberies_in_boston = pd.read_csv("../input/monthly-boston-armed-robberies-j.csv")

series = robberies_in_boston["Count"]
train, test, val = series[:80], series[80:100], series[100:]
def mse(X):

    alpha, beta = X

    result = double_ema_with_preds(train, alpha, beta, len(test))

    predictions = result[-len(test):]

    error = mean_squared_error(predictions, test)

    return error
opt = minimize(mse, x0=[0,0], method="L-BFGS-B", bounds = ((0, 1), (0, 1)))
alpha_opt, beta_opt = opt.x

print(opt)
def plot_dema(alpha, beta, ser=robberies_in_boston["Count"], ser_to_plot=robberies_in_boston["Count"], n_preds=24):

    dema = double_ema_with_preds(ser, alpha, beta, n_preds)

    with plt.style.context('bmh'):

        plt.figure(figsize=(14, 8))

        plt.plot(ser_to_plot, color='blue',label='original')

        plt.plot(dema, color='red', linewidth='4', label='DEMA')

        plt.title("alpha={}, beta={}".format(alpha, beta))

        plt.legend()
plot_dema(alpha_opt, beta_opt, ser=train, ser_to_plot=series[:100], n_preds=len(test))
import pandas as pd

import numpy as np
dowjones_closing = pd.read_csv("../input/weekly-closings-of-the-dowjones-.csv")

female_births = pd.read_csv("../input/daily-total-female-births-in-cal.csv")
all_series = {

    "Weekly closings of the Dow-Jones industrial average": dowjones_closing["Close"],

    "Daily total female births in California": female_births["Count"]

}
from statsmodels.tsa.stattools import adfuller

def test_stationarity(timeseries):

    print('Results of Dickey-Fuller Test:')

    dftest = adfuller(timeseries, autolag='AIC')

    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])

    for [key, value] in dftest[4].items():

        dfoutput['Critical Value (%s)' % key] = value

    print(dfoutput)
test_stationarity(dowjones_closing["Close"])
with plt.style.context('bmh'):

    plt.plot(dowjones_closing["Close"], color='blue') # не стационарный ряд
def double_ema(series, alpha, beta):

    result = [series[0]]

    level, trend = series[0], series[1] - series[0]

    for n in range(1, len(series)):

        value = series[n]

        last_level, level = level, alpha*value + (1-alpha)*(level+trend)

        trend = beta*(level-last_level) + (1-beta)*trend

        result.append(level+trend)

    return pd.Series(result)
def plot_dema(alpha, beta):

    dema = double_ema(dowjones_closing["Close"], alpha, beta)

    with plt.style.context('bmh'):

        plt.plot(dowjones_closing["Close"], color='blue',label='original')

        plt.plot(dema, color='red', linewidth='4', label='DEMA')

        plt.title("alpha={}, beta={}".format(alpha, beta))

        plt.legend()
plot_dema(0.6, 0.2)
test_stationarity(female_births["Count"])
with plt.style.context('bmh'):

    plt.plot(female_births["Count"], color='blue') # Стационарный ряд
def exponential_moving_average(series, alpha):

    result = [series[0]]

    for n in range(1, len(series)):

        result.append(alpha * series[n] + (1 - alpha) * result[n-1])

    return pd.Series(result)
ema = exponential_moving_average(female_births["Count"], 0.1)
with plt.style.context('bmh'):

    plt.plot(female_births["Count"], color='blue',label='original')

    plt.plot(ema, color='red', linewidth='4', label='DEMA')