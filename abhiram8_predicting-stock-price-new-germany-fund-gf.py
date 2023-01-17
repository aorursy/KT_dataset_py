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
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

sns.set()



from sklearn.metrics import r2_score, median_absolute_error, mean_absolute_error

from sklearn.metrics import median_absolute_error, mean_squared_error, mean_squared_log_error



from scipy.optimize import minimize

import statsmodels.tsa.api as smt

import statsmodels.api as sm



from tqdm import tqdm_notebook



from itertools import product



def mean_absolute_percentage_error(y_true, y_pred):

    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100



import warnings

warnings.filterwarnings('ignore')



%matplotlib inline



DATAPATH = 'https://raw.githubusercontent.com/marcopeix/stock-prediction/master/data/stock_prices_sample.csv'



#data = pd.read_csv(DATAPATH)

#data.head(10)
data = pd.read_csv(DATAPATH, index_col=['DATE'], parse_dates=['DATE'])

data.head(10)
data = data[data.TICKER != 'GEF']

data = data[data.TYPE != 'Intraday']



drop_cols = ['SPLIT_RATIO', 'EX_DIVIDEND', 'ADJ_FACTOR', 'ADJ_VOLUME', 'ADJ_CLOSE', 'ADJ_LOW', 'ADJ_HIGH', 'ADJ_OPEN', 'VOLUME', 'FREQUENCY', 'TYPE', 'FIGI']



data.drop(drop_cols, axis=1, inplace=True)



data.head()
# Plot closing price



plt.figure(figsize=(17, 8))

plt.plot(data.CLOSE)

plt.title('Closing price of New Germany Fund Inc (GF)')

plt.ylabel('Closing price ($)')

plt.xlabel('Trading day')

plt.grid(False)

plt.show()
def plot_moving_average(series, window, plot_intervals=False, scale=1.96):



    rolling_mean = series.rolling(window=window).mean()

    

    plt.figure(figsize=(17,8))

    plt.title('Moving average\n window size = {}'.format(window))

    plt.plot(rolling_mean, 'g', label='Rolling mean trend')

    

    #Plot confidence intervals for smoothed values

    if plot_intervals:

        mae = mean_absolute_error(series[window:], rolling_mean[window:])

        deviation = np.std(series[window:] - rolling_mean[window:])

        lower_bound = rolling_mean - (mae + scale * deviation)

        upper_bound = rolling_mean + (mae + scale * deviation)

        plt.plot(upper_bound, 'r--', label='Upper bound / Lower bound')

        plt.plot(lower_bound, 'r--')

            

    plt.plot(series[window:], label='Actual values')

    plt.legend(loc='best')

    plt.grid(True)

    

#Smooth by the previous 5 days (by week)

plot_moving_average(data.CLOSE, 5)



#Smooth by the previous month (30 days)

plot_moving_average(data.CLOSE, 30)



#Smooth by previous quarter (90 days)

plot_moving_average(data.CLOSE, 90, plot_intervals=True)
def exponential_smoothing(series, alpha):



    result = [series[0]] # first value is same as series

    for n in range(1, len(series)):

        result.append(alpha * series[n] + (1 - alpha) * result[n-1])

    return result

  

def plot_exponential_smoothing(series, alphas):

 

    plt.figure(figsize=(17, 8))

    for alpha in alphas:

        plt.plot(exponential_smoothing(series, alpha), label="Alpha {}".format(alpha))

    plt.plot(series.values, "c", label = "Actual")

    plt.legend(loc="best")

    plt.axis('tight')

    plt.title("Exponential Smoothing")

    plt.grid(True);



plot_exponential_smoothing(data.CLOSE, [0.05, 0.3])
def double_exponential_smoothing(series, alpha, beta):



    result = [series[0]]

    for n in range(1, len(series)+1):

        if n == 1:

            level, trend = series[0], series[1] - series[0]

        if n >= len(series): # forecasting

            value = result[-1]

        else:

            value = series[n]

        last_level, level = level, alpha * value + (1 - alpha) * (level + trend)

        trend = beta * (level - last_level) + (1 - beta) * trend

        result.append(level + trend)

    return result



def plot_double_exponential_smoothing(series, alphas, betas):

     

    plt.figure(figsize=(17, 8))

    for alpha in alphas:

        for beta in betas:

            plt.plot(double_exponential_smoothing(series, alpha, beta), label="Alpha {}, beta {}".format(alpha, beta))

    plt.plot(series.values, label = "Actual")

    plt.legend(loc="best")

    plt.axis('tight')

    plt.title("Double Exponential Smoothing")

    plt.grid(True)

    

plot_double_exponential_smoothing(data.CLOSE, alphas=[0.9, 0.02], betas=[0.9, 0.02])
def tsplot(y, lags=None, figsize=(12, 7), syle='bmh'):

    

    if not isinstance(y, pd.Series):

        y = pd.Series(y)

        

    with plt.style.context(style='bmh'):

        fig = plt.figure(figsize=figsize)

        layout = (2,2)

        ts_ax = plt.subplot2grid(layout, (0,0), colspan=2)

        acf_ax = plt.subplot2grid(layout, (1,0))

        pacf_ax = plt.subplot2grid(layout, (1,1))

        

        y.plot(ax=ts_ax)

        p_value = sm.tsa.stattools.adfuller(y)[1]

        ts_ax.set_title('Time Series Analysis Plots\n Dickey-Fuller: p={0:.5f}'.format(p_value))

        smt.graphics.plot_acf(y, lags=lags, ax=acf_ax)

        smt.graphics.plot_pacf(y, lags=lags, ax=pacf_ax)

        plt.tight_layout()

        

tsplot(data.CLOSE, lags=30)



# Take the first difference to remove to make the process stationary

data_diff = data.CLOSE - data.CLOSE.shift(1)



tsplot(data_diff[1:], lags=30)
#Set initial values and some bounds

ps = range(0, 5)

d = 1

qs = range(0, 5)

Ps = range(0, 5)

D = 1

Qs = range(0, 5)

s = 5



#Create a list with all possible combinations of parameters

parameters = product(ps, qs, Ps, Qs)

parameters_list = list(parameters)

len(parameters_list)



# Train many SARIMA models to find the best set of parameters

def optimize_SARIMA(parameters_list, d, D, s):

    """

        Return dataframe with parameters and corresponding AIC

        

        parameters_list - list with (p, q, P, Q) tuples

        d - integration order

        D - seasonal integration order

        s - length of season

    """

    

    results = []

    best_aic = float('inf')

    

    for param in tqdm_notebook(parameters_list):

        try: model = sm.tsa.statespace.SARIMAX(data.CLOSE, order=(param[0], d, param[1]),

                                               seasonal_order=(param[2], D, param[3], s)).fit(disp=-1)

        except:

            continue

            

        aic = model.aic

        

        #Save best model, AIC and parameters

        if aic < best_aic:

            best_model = model

            best_aic = aic

            best_param = param

        results.append([param, model.aic])

        

    result_table = pd.DataFrame(results)

    result_table.columns = ['parameters', 'aic']

    #Sort in ascending order, lower AIC is better

    result_table = result_table.sort_values(by='aic', ascending=True).reset_index(drop=True)

    

    return result_table



result_table = optimize_SARIMA(parameters_list, d, D, s)



#Set parameters that give the lowest AIC (Akaike Information Criteria)

p, q, P, Q = result_table.parameters[0]



best_model = sm.tsa.statespace.SARIMAX(data.CLOSE, order=(p, d, q),

                                       seasonal_order=(P, D, Q, s)).fit(disp=-1)



print(best_model.summary())
# Make a dataframe containing actual and predicted prices

comparison = pd.DataFrame({'actual': [18.93, 19.23, 19.08, 19.17, 19.11, 19.12],

                          'predicted': [18.96, 18.97, 18.96, 18.92, 18.94, 18.92]}, 

                          index = pd.date_range(start='2018-06-05', periods=6,))





#Plot predicted vs actual price



plt.figure(figsize=(17, 8))

plt.plot(comparison.actual)

plt.plot(comparison.predicted)

plt.title('Predicted closing price of New Germany Fund Inc (GF)')

plt.ylabel('Closing price ($)')

plt.xlabel('Trading day')

plt.legend(loc='best')

plt.grid(False)

plt.show()