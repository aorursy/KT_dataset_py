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
# Major libraries related to Data handling, Vis and statistics

import numpy as np

import pandas as pd

from scipy import stats

import matplotlib.pyplot as plt

import datetime as dt

import seaborn as sns

from scipy.stats import normaltest, skew

from sklearn.preprocessing import StandardScaler

from matplotlib.colors import ListedColormap

from sklearn.metrics import r2_score, median_absolute_error, mean_absolute_error

from sklearn.metrics import median_absolute_error, mean_squared_error, mean_squared_log_error

from statsmodels.tsa.stattools import pacf

from statsmodels.graphics.tsaplots import plot_pacf, plot_acf

from statsmodels.tsa.seasonal import seasonal_decompose

from statsmodels.tsa.stattools import adfuller

from statsmodels.tsa.arima_model import ARIMA, ARMA

from statsmodels.tsa import stattools



from IPython.display import set_matplotlib_formats 

plt.style.use('ggplot')

sns.set_style('whitegrid')

sns.set(font_scale=1.5)

%config InlineBackend.figure_format = 'retina'



import warnings

warnings.filterwarnings("ignore")



# Pallets used for visualizations

color= "Spectral"

color_plt = ListedColormap(sns.color_palette(color).as_hex())

color_hist = 'teal'

BOLD = '\033[1m'

END = '\033[0m'
stocks = pd.read_csv('/kaggle/input/saudi-stock-exchange-tadawul/Tadawul_stcks.csv')

stocks_2 = pd.read_csv('/kaggle/input/saudi-stock-exchange-tadawul/Tadawul_stcks_23_4.csv')

stocks = stocks_2.append(stocks,ignore_index=True)

stocks.rename(columns={'trading_name ': 'trading_name', 'volume_traded ': 'volume_traded','no_trades ':'no_trades'}, inplace=True)

stocks.head()
stocks[stocks['sectoer']=='Health Care']['trading_name'].unique()
health_care = stocks[stocks['sectoer']=='Health Care']

health_care['date']= pd.to_datetime(health_care['date'])

health_care.sort_values('date', inplace=True)

health_care = health_care.set_index('date')

health_care.head()
health_care.info()
health_care.isna().sum() # it's ok that open, high, and low has some missing values, we're not going to us them anyways.
plt.figure(figsize=(17, 6))

sns.lineplot(x=health_care.index, y="close", hue="trading_name", markers=True, data=health_care)

plt.title('Closing price of Saudi Stocks in the Healthcare Sector')

plt.ylabel('Closing price ($)')

plt.xlabel('Year')

plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),

          fancybox=True, shadow=True, ncol=5)

plt.grid(False)

plt.show()
stocks[stocks['trading_name']=='CHEMICAL'].tail(1)
stocks[stocks['trading_name']=='SPIMACO'].tail(1)
covid_19 = pd.read_csv('/kaggle/input/saudi-covid19-from-11-to-184/saudi_covid-19.csv', index_col='Date', parse_dates=['Date'])

covid_19.drop('Unnamed: 0',axis=1, inplace=True)

covid_19 = covid_19 [['ConfirmedCases', 'ConfirmedDeaths', 'StringencyIndex']]

covid_19.describe().T
# The date that Saudi Arabia reported the first case of covid-19

covid_19[covid_19['ConfirmedCases']==1].head(1)
# The date that Saudi Arabia closed everything and had the lockdown.

covid_19[covid_19['StringencyIndex']==covid_19.max()['StringencyIndex']].head(1)
health_care_2020 = health_care.loc[health_care.index>'2020-01-01']

plt.figure(figsize=(17, 6))

sns.lineplot(x=health_care_2020.index, y="close", hue="trading_name", markers=True, data=health_care_2020)

plt.title('Closing price of Saudi Stocks in the Healthcare Sector in 2020 and COVID-19 Analysis')

plt.ylabel('Closing price ($)')

plt.xlabel('Year')



plt.axvline(x= dt.datetime(2020,3,3))

plt.text(x=dt.datetime(2020,3,1),y=20,s='First case of Covid-19 reported',rotation=90)

plt.axvline(x=dt.datetime(2020,3,23))

plt.text(x=dt.datetime(2020,3,21),y=20,s='First day of lockdown',rotation=90)





plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),

          fancybox=True, shadow=True, ncol=5)

plt.grid(True)

plt.show()
CHEMICAL_df = stocks[stocks['trading_name']=='CHEMICAL']

CHEMICAL_df['date']= pd.to_datetime(CHEMICAL_df['date'])

CHEMICAL_df.sort_values('date', inplace=True)

CHEMICAL_df = CHEMICAL_df.set_index('date')

CHEMICAL_df.drop('symbol',axis=1).describe().T
SPIMACO_df = stocks[stocks['trading_name']=='SPIMACO']

SPIMACO_df['date']= pd.to_datetime(SPIMACO_df['date'])

SPIMACO_df.sort_values('date', inplace=True)

SPIMACO_df = SPIMACO_df.set_index('date')

SPIMACO_df.drop('symbol',axis=1).describe().T
plt.figure(figsize=(17, 4))

plt.plot(CHEMICAL_df['close'])

plt.title('Closing price of Saudi Chemical Co.')

plt.ylabel('Closing price ($)')

plt.xlabel('Year')

plt.grid(False)

plt.show()



plt.figure(figsize=(17, 4))

plt.plot(SPIMACO_df['close'])

plt.title('Closing price of Saudi Pharmaceutical Industries & Medical Appliances Corporation')

plt.ylabel('Closing price ($)')

plt.xlabel('Year')

plt.grid(False)

plt.show()
def mean_absolute_percentage_error(y_true, y_pred):

    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
def plot_moving_average(series, window, plot_intervals=False, scale=1.96):



    rolling_mean = series.rolling(window = window).mean()

    resample_mean = series.resample('Y').mean()

    

    plt.figure(figsize=(17,6))

    plt.title('Moving Average for ')

    plt.plot(series[window:], label='Actual values')

    plt.plot(rolling_mean, 'g', label='Rolling mean trend window size = {}'.format(window) , linewidth=4.0)

    plt.plot(resample_mean, 'black', label='Yearly Resample mean trend')

    

    #Plot confidence intervals for smoothed values

    if plot_intervals:

        mae = mean_absolute_error(series[window:], rolling_mean[window:])

        deviation = np.std(series[window:] - rolling_mean[window:])

        lower_bound = rolling_mean - (mae + scale * deviation)

        upper_bound = rolling_mean + (mae + scale * deviation)

        plt.plot(upper_bound, 'r--', label='Upper bound / Lower bound')

        plt.plot(lower_bound, 'r--')

            

    

    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),

          fancybox=True, shadow=True, ncol=5)    

    plt.grid(True)
#Smooth by previous quarter (90 days)

plot_moving_average(CHEMICAL_df['close'], 100, plot_intervals=True)

plt.ylabel('Closing price ($)')

plt.title ('Moving Average for Closing price of Saudi Chemical Co.');



#Smooth by previous quarter (90 days)

plot_moving_average(SPIMACO_df['close'], 100, plot_intervals=True)

plt.ylabel('Closing price ($)')

plt.title ('Moving Average for Closing price of Saudi Pharmaceutical Industries & Medical Appliances Corporation');
# I think that there is something wrong because I don't think that the data looks stationary, 

# but the p-value is smaller than 0.05!

# why?

print(BOLD + 'CHEMICAL' +END)

result = adfuller(CHEMICAL_df['close'].dropna())

print(f'ADF Statistic: {result[0]}')

print(f'p-value: {result[1]}')

print('p_value < 0.05 , that means the series is stationary\n')

for key, value in result[4].items():

    print('Critial Values:')

    print(f'   {key}, {value}')

    

print('\n' + BOLD + 'SPIMACO' +END)

result = adfuller(SPIMACO_df['close'].dropna())

print(f'ADF Statistic: {result[0]}')

print(f'p-value: {result[1]}')

print('p_value < 0.05 , that means the series is stationary\n')

for key, value in result[4].items():

    print('Critial Values:')

    print(f'   {key}, {value}')    
CHEMICAL_df['close_diff'] = CHEMICAL_df['close'].diff()

plot_moving_average(CHEMICAL_df['close_diff'], 100, plot_intervals=True)

plt.ylabel('Difference in closing price ($)')

plt.title ('Difference in Moving Average for Closing price of Saudi Chemical Co.');



SPIMACO_df['close_diff'] = SPIMACO_df['close'].diff()

plot_moving_average(SPIMACO_df['close_diff'], 100, plot_intervals=True)

plt.ylabel('Difference in closing price ($)')

plt.title ('Difference in Moving Average for Closing price of Saudi Pharmaceutical Industries & Medical Appliances Corporation');
plt.figure(figsize=(10, 6))

sns.distplot(CHEMICAL_df['close_diff'].dropna())

plt.title('Distribution of Difference in closing price of Saudi Chemical Co. (%)')

plt.ylabel('Frequency')

plt.xlabel('Difference in closing price (%)')

plt.show()



plt.figure(figsize=(10, 6))

sns.distplot(SPIMACO_df['close_diff'].dropna())

plt.title('Distribution of Difference in closing price of Saudi Pharmaceutical Industries & Medical Appliances Corporation (%)')

plt.ylabel('Frequency')

plt.xlabel('Difference in closing price (%)')

plt.show()
print(BOLD + 'CHEMICAL' +END)

result = adfuller(CHEMICAL_df['close_diff'].dropna())

print(f'ADF Statistic: {result[0]}')

print(f'p-value: {result[1]}')

print('p_value < 0.05 , that means the series is stationary\n')

for key, value in result[4].items():

    print('Critial Values:')

    print(f'   {key}, {value}')



print('\n'+ BOLD + 'SPIMACO' +END)

result = adfuller(SPIMACO_df['close_diff'].dropna())

print(f'ADF Statistic: {result[0]}')

print(f'p-value: {result[1]}')

print('p_value < 0.05 , that means the series is stationary\n')

for key, value in result[4].items():

    print('Critial Values:')

    print(f'   {key}, {value}')    

    
CHEMICAL_df['close'].autocorr()
SPIMACO_df['close'].autocorr()
def autocorr_plots(y, lags=None):

    fig, ax = plt.subplots(ncols=2, figsize=(15, 4), sharey=True)

    plot_acf(y, lags=lags, ax=ax[0])

    plot_pacf(y, lags=lags, ax=ax[1])

    return fig, ax
fig, ax = autocorr_plots(CHEMICAL_df['close'],lags=30)
fig, ax = autocorr_plots(SPIMACO_df['close'],lags=30)
corr_CHEMICAL_df = CHEMICAL_df.drop('symbol', axis=1).corr()

fig, axs = plt.subplots(figsize = (13, 10)) 

mask = np.triu(np.ones_like(corr_CHEMICAL_df, dtype = np.bool))

sns.heatmap(corr_CHEMICAL_df, ax = axs, mask = mask, cmap = sns.diverging_palette(180, 10, as_cmap = True))

plt.title('Correlation of the stock prices in Saudi Chemical Co.')



# fix for mpl bug that cuts off top/bottom of seaborn viz

b, t = plt.ylim() # discover the values for bottom and top

b += 0.5 # Add 0.5 to the bottom

t -= 0.5 # Subtract 0.5 from the top

plt.ylim(b, t) # update the ylim(bottom, top) values

plt.show()
corr_SPIMACO_df = SPIMACO_df.drop('symbol', axis=1).corr()

fig, axs = plt.subplots(figsize = (13, 10)) 

mask = np.triu(np.ones_like(corr_SPIMACO_df, dtype = np.bool))

sns.heatmap(corr_SPIMACO_df, ax = axs, mask = mask, cmap = sns.diverging_palette(180, 10, as_cmap = True))

plt.title('Correlation of the stock pricess in Saudi Pharmaceutical Industries & Medical Appliances Corporation')



# fix for mpl bug that cuts off top/bottom of seaborn viz

b, t = plt.ylim() # discover the values for bottom and top

b += 0.5 # Add 0.5 to the bottom

t -= 0.5 # Subtract 0.5 from the top

plt.ylim(b, t) # update the ylim(bottom, top) values

plt.show()
# evaluate an ARIMA model for a given order (p,d,q)

def evaluate_arima_model(X, arima_order):

    # prepare training dataset

    train_size = int(len(X) * 0.66)

    train, test = X[0:train_size], X[train_size:]

    history = [x for x in train]

    # make predictions

    predictions = list()

    for t in range(len(test)):

        model = ARIMA(history, order=arima_order)

        model_fit = model.fit(disp=0)

        yhat = model_fit.forecast()[0]

        predictions.append(yhat)

        history.append(test[t])

    # calculate out of sample error

    error = mean_squared_error(test, predictions)

    return error
# evaluate combinations of p, d and q values for an ARIMA model

def evaluate_models(dataset, p_values, d_values, q_values):

    dataset = dataset.astype('float32')

    best_score, best_cfg = float("inf"), None

    for p in p_values:

        for d in d_values:

            for q in q_values:

                order = (p,d,q)

                try:

                    mse = evaluate_arima_model(dataset, order)

                    if mse < best_score:

                        best_score, best_cfg = mse, order

                    print('ARIMA%s MSE=%.3f' % (order,mse))

                except:

                    continue

    print('Best ARIMA%s MSE=%.3f' % (best_cfg, best_score))
CHEMICAL_df.index.max(), CHEMICAL_df.index.min()
# mid year 

int((CHEMICAL_df.index.max().year + CHEMICAL_df.index.min().year) /2)
df_train = CHEMICAL_df.loc[:'2011']

df_test = CHEMICAL_df.loc['2012':'2019']



# train 

close_train = df_train['close']

udiff_train = close_train.diff().dropna()



#test

close_test = df_test['close']

udiff_test = close_test.diff().dropna()
df_train.shape , df_test.shape
# Plot the train and test sets on the axis ax

fig, ax = plt.subplots(figsize=(10, 6))

df_train['close'].plot(ax=ax)

df_test['close'].plot(ax=ax)

plt.title('Closing price of Saudi Chemical Co.')

plt.ylabel('Closing price ($)')

plt.xlabel('Year')

plt.grid(False)

plt.show()
# Plot the train and test sets on the axis ax

fig, ax = plt.subplots(figsize=(10, 6))

udiff_train.plot(ax=ax)

udiff_test.plot(ax=ax)

plt.title('Difference in Closing price of Saudi Chemical Co.')

plt.ylabel('Difference Closing price ($)')

plt.xlabel('Year')

plt.grid(False)

plt.show()
#find the optimal parameters using AIC & BIC

auto_select = stattools.arma_order_select_ic(df_train['close'], max_ar=5, max_ma=5, ic=['aic', 'bic'])



plt.subplots(figsize = (10,8))

sns.heatmap(auto_select['aic'],  cmap = sns.diverging_palette(180, 10, as_cmap = True),square=True, fmt='.1f')

plt.title('AIC')



# fix for mpl bug that cuts off top/bottom of seaborn viz

b, t = plt.ylim() # discover the values for bottom and top

b += 0.5 # Add 0.5 to the bottom

t -= 0.5 # Subtract 0.5 from the top

plt.ylim(b, t) # update the ylim(bottom, top) values

plt.show()



plt.subplots(figsize = (10,8))

sns.heatmap(auto_select['bic'],  cmap = sns.diverging_palette(180, 10, as_cmap = True),square=True, fmt='.1f')

plt.title('BIC')



# fix for mpl bug that cuts off top/bottom of seaborn viz

b, t = plt.ylim() # discover the values for bottom and top

b += 0.5 # Add 0.5 to the bottom

t -= 0.5 # Subtract 0.5 from the top

plt.ylim(b, t) # update the ylim(bottom, top) values

plt.show()
model = ARIMA(close_train,order=(2,0,0))

res = model.fit()

res.summary()
# plot our prediction for train data

fig, ax = plt.subplots(figsize=(10, 6))

close_train.plot(legend = True)

res.fittedvalues.rename("Trainset Predictions").plot(legend = True)

plt.title('Actual vs. Prediction in Trainset Closing price of Saudi Chemical Co.')

plt.ylabel('Difference Closing price ($)')

plt.xlabel('Year')

plt.grid(False)

plt.show()
start = len(udiff_train) 

end = len(udiff_train) + len(udiff_test) 

  

# Predictions for the test set 

predictions = res.predict(start, end).rename("Testset Predictions") 

predictions = pd.DataFrame(predictions)

predictions.set_index(df_test.index,inplace=True)



fig, ax = plt.subplots(figsize=(10, 6))



close_test.plot(legend = True, ax = ax)

predictions.plot(legend = True, ax = ax) 



plt.title('Actual vs. Prediction in Testset Closing price of Saudi Chemical Co.')

plt.ylabel('Difference Closing price ($)')

plt.xlabel('Year')

plt.grid(False)

plt.show()
SPIMACO_df.index.max(), SPIMACO_df.index.min()
# mid year 

int((SPIMACO_df.index.max().year + SPIMACO_df.index.min().year) /2)
df_train = SPIMACO_df.loc[:'2011']

df_test = SPIMACO_df.loc['2012':'2019']



# train 

close_train = df_train['close']

udiff_train = close_train.diff().dropna()



#test

close_test = df_test['close']

udiff_test = close_test.diff().dropna()
df_train.shape , df_test.shape
# Plot the train and test sets on the axis ax

fig, ax = plt.subplots(figsize=(10, 6))

df_train['close'].plot(ax=ax)

df_test['close'].plot(ax=ax)

plt.title('Closing price of Saudi Pharmaceutical Industries & Medical Appliances Corporation')

plt.ylabel('Closing price ($)')

plt.xlabel('Year')

plt.grid(False)

plt.show()
# Plot the train and test sets on the axis ax

fig, ax = plt.subplots(figsize=(10, 6))

udiff_train.plot(ax=ax)

udiff_test.plot(ax=ax)

plt.title('Difference in Closing price of Saudi Pharmaceutical Industries & Medical Appliances Corporation')

plt.ylabel('Difference Closing price ($)')

plt.xlabel('Year')

plt.grid(False)

plt.show()
#find the optimal parameters using AIC & BIC

auto_select = stattools.arma_order_select_ic(df_train['close'], max_ar=5, max_ma=5, ic=['aic', 'bic'])



plt.subplots(figsize = (10,8))

sns.heatmap(auto_select['aic'],  cmap = sns.diverging_palette(180, 10, as_cmap = True),square=True, fmt='.1f')

plt.title('AIC')



# fix for mpl bug that cuts off top/bottom of seaborn viz

b, t = plt.ylim() # discover the values for bottom and top

b += 0.5 # Add 0.5 to the bottom

t -= 0.5 # Subtract 0.5 from the top

plt.ylim(b, t) # update the ylim(bottom, top) values

plt.show()



plt.subplots(figsize = (10,8))

sns.heatmap(auto_select['bic'],  cmap = sns.diverging_palette(180, 10, as_cmap = True),square=True, fmt='.1f')

plt.title('BIC')



# fix for mpl bug that cuts off top/bottom of seaborn viz

b, t = plt.ylim() # discover the values for bottom and top

b += 0.5 # Add 0.5 to the bottom

t -= 0.5 # Subtract 0.5 from the top

plt.ylim(b, t) # update the ylim(bottom, top) values

plt.show()
model = ARIMA(close_train,order=(1,0,1))

res = model.fit()

res.summary()
# plot our prediction for train data

fig, ax = plt.subplots(figsize=(10, 6))

close_train.plot(legend = True)

res.fittedvalues.rename("Trainset Predictions").plot(legend = True)

plt.title('Actual vs. Prediction in Trainset Closing price ofSaudi Pharmaceutical Industries & Medical Appliances Corporation')

plt.ylabel('Difference Closing price ($)')

plt.xlabel('Year')

plt.grid(False)

plt.show()
start = len(udiff_train) 

end = len(udiff_train) + len(udiff_test) 

  

# Predictions for the test set 

predictions = res.predict(start, end).rename("Testset Predictions") 

predictions = pd.DataFrame(predictions)

predictions.set_index(df_test.index,inplace=True)



fig, ax = plt.subplots(figsize=(10, 6))



close_test.plot(legend = True, ax = ax)

predictions.plot(legend = True, ax = ax) 



plt.title('Actual vs. Prediction in Testset Closing price of Saudi Pharmaceutical Industries & Medical Appliances Corporation')

plt.ylabel('Difference Closing price ($)')

plt.xlabel('Year')

plt.grid(False)

plt.show()