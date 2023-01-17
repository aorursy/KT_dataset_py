# import statement

import pandas as pd

import matplotlib.pyplot as plt

import matplotlib as mpl

import numpy as np

import seaborn as sns
# importing data

df = pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/a10.csv', parse_dates=['date'])

df.head()
# comparing trend and seasonality over the years



# data prep

df['year'] = [d.year for d in df.date]

df['month'] = [d.month for d in df.date]

years = df['year'].unique()



# color prep

np.random.seed(100)

mycolors = np.random.choice(list(mpl.colors.XKCD_COLORS.keys()), len(years.tolist()), replace=False)



# Draw Plot

plt.figure(figsize=(16,12), dpi= 80)

for i, y in enumerate(years):

    if i > 0:        

        # loc is based on condition

        plt.plot('month', 'value', data=df.loc[df.year==y, :], color=mycolors[i], label=y)

        

        # text for graph

        plt.text(df.loc[df.year==y, :].shape[0]-.9, df.loc[df.year==y, 'value'][-1:].values[0], y, fontsize=12, 

                 color=mycolors[i])



# Decoration



# gca = get current axis

plt.gca().set(xlim=(-0.3, 11), ylim=(2, 30), ylabel='$Drug Sales$', xlabel='$Month$')

plt.yticks(fontsize=12, alpha=.7)

plt.title("Seasonal Plot of Drug Sales Time Series", fontsize=20)

plt.show()
# Draw Plot



# plt.subplot(nrows, ncols)

fig, axes = plt.subplots(1, 2, figsize=(20,7), dpi= 80)

sns.boxplot(x='year', y='value', data=df, ax=axes[0])

sns.boxplot(x='month', y='value', data=df.loc[~df.year.isin([1991, 2008]), :])



# Set Title

axes[0].set_title('Year-wise Box Plot\n(The Trend)', fontsize=18); 

axes[1].set_title('Month-wise Box Plot\n(The Seasonality)', fontsize=18)

plt.show()
fig, axes = plt.subplots(1,3, figsize=(20,4), dpi=100)

pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/guinearice.csv', 

            parse_dates=['date'], 

            index_col='date').plot(title='Trend Only', legend=False, ax=axes[0])



pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/sunspotarea.csv', 

            parse_dates=['date'], 

            index_col='date').plot(title='Seasonality Only', legend=False, ax=axes[1])



pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/AirPassengers.csv', 

            parse_dates=['date'], 

            index_col='date').plot(title='Trend and Seasonality', legend=False, ax=axes[2])
# using statsmodel to decompose a time series to its components to find its type



from statsmodels.tsa.seasonal import seasonal_decompose

from dateutil.parser import parse



# Import Data

df = pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/a10.csv', 

                 parse_dates=['date'], index_col='date')



# Multiplicative Decomposition 

result_mul = seasonal_decompose(df['value'], model='multiplicative', extrapolate_trend='freq')



# Additive Decomposition

result_add = seasonal_decompose(df['value'], model='additive', extrapolate_trend='freq')



# Plot

#plt.rcParams.update({'figure.figsize': (10,10)})

result_mul.plot().suptitle('Multiplicative Decompose', fontsize=22)

result_add.plot().suptitle('Additive Decompose', fontsize=22)

plt.show()
# Extract the Components ----

# Actual Values = Product of (Seasonal * Trend * Resid)

df_reconstructed = pd.concat([result_mul.seasonal, result_mul.trend, result_mul.resid, 

                              result_mul.observed], axis=1)

df_reconstructed.columns = ['seas', 'trend', 'resid', 'actual_values']

df_reconstructed.head()
df_reconstructed['seas'].iloc[0] * df_reconstructed['trend'].iloc[0] * df_reconstructed['resid'].iloc[0]



# as expected, the actual_values = seas * trend * resid
from statsmodels.tsa.stattools import adfuller, kpss

df = pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/a10.csv', 

                 parse_dates=['date'])



# ADF Test

result = adfuller(df.value.values, autolag='AIC')

print(f'ADF Statistic: {result[0]}')

print(f'p-value: {result[1]}')

for key, value in result[4].items():

    print('Critial Values:')

    print(f'   {key}, {value}')



# KPSS Test

result = kpss(df.value.values, regression='c')

print('\nKPSS Statistic: %f' % result[0])

print('p-value: %f' % result[1])

for key, value in result[3].items():

    print('Critial Values:')

    print(f'   {key}, {value}')
# Using statmodels: Subtracting the Trend Component.

from statsmodels.tsa.seasonal import seasonal_decompose



df = pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/a10.csv', 

                 parse_dates=['date'], index_col='date')



result_mul = seasonal_decompose(df['value'], model='multiplicative', extrapolate_trend='freq')



detrended = df.value.values - result_mul.trend

print (df.value.values, result_mul.trend)

plt.plot(detrended)

plt.title('Drug Sales detrended by subtracting the trend component', fontsize=16)
from pandas.plotting import autocorrelation_plot

df = pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/a10.csv')



# Draw Plot

plt.rcParams.update({'figure.figsize':(9,5), 'figure.dpi':120})

autocorrelation_plot(df.value.tolist())
# Subtracting the Trend Component.

df = pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/a10.csv', 

                 parse_dates=['date'], index_col='date')



# Time Series Decomposition

result_mul = seasonal_decompose(df['value'], model='multiplicative', extrapolate_trend='freq')



# Deseasonalize

deseasonalized = df.value.values / result_mul.seasonal



# Plot

plt.plot(deseasonalized)

plt.title('Drug Sales Deseasonalized', fontsize=16)

plt.plot()
# End to end example using AirPassenger Dataset



# Loading data, ensuring it is indexed by the date.

df_air = pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/AirPassengers.csv', 

            parse_dates=['date'], index_col='date')



# Plotting data.

df_air.plot()
# The data clearly has an upward trend as well as a seasonality. Extracting the time 

# series components using statsmodel library's seasonal_decompose.



from statsmodels.tsa.seasonal import seasonal_decompose



# The series could be multiplicative or additive.

result_mul = seasonal_decompose(df_air['value'], model='multiplicative', extrapolate_trend='freq')

result_add = seasonal_decompose(df_air['value'], model='additive', extrapolate_trend='freq')



result_mul.plot().suptitle('Multiplicative Decompose', fontsize=22)

result_add.plot().suptitle('Additive Decompose', fontsize=22)

plt.show()
# From the residual plots we see that the time series is multiplicative in nature. We need to 

# detrend and deseasonalize the series. Before we do that, lets check if the series is stationary or not.



# Performing AD Fuller Test. If the p-value is much smaller than the critical value then we can reject

# the null hypothesis (that the series is stationary).



from statsmodels.tsa.stattools import adfuller



# ADF Test

result = adfuller(df_air.value.values, autolag='AIC')

print(f'ADF Statistic: {result[0]}')

print(f'p-value: {result[1]}')

for key, value in result[4].items():

    print('Critial Values:')

    print(f'   {key}, {value}')
# Detrending by subtracting line of best bit.

detrended_df = df_air.value.values - result_mul.trend

plt.plot(detrended_df)
# Before deseasonalizing, we can infact empirically see that there is a seasonality but lets confirm it

# statistically as well using autocorrelation plots



from pandas.plotting import autocorrelation_plot



# Draw Plot

plt.rcParams.update({'figure.figsize':(9,5), 'figure.dpi':120})

autocorrelation_plot(detrended.values.tolist())
# Detrending the series by dividing by seasonal component given by decomposition.



deseasoned = df_air.value.values / result_mul.seasonal

final_df = deseasoned.values - result_mul.trend

plt.plot(final_df)
df_before = df_air.head(100)



from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt



# using alpha value as 0.2

# the optimized = False is with reference to log likelyhood, I still havent understood its usecase here

# the rename part is just for the graph's legend

fit1 = SimpleExpSmoothing(df_before).fit(smoothing_level=0.2,optimized=False)

fcast1 = fit1.forecast(3).rename(r'$\alpha=0.2$')



# using alpha value as 0.6

fit2 = SimpleExpSmoothing(df_before).fit(smoothing_level=0.6,optimized=False)

fcast2 = fit2.forecast(3).rename(r'$\alpha=0.6$')



# letting the model decide the best alpha value

fit3 = SimpleExpSmoothing(df_before).fit()

fcast3 = fit3.forecast(2).rename(r'$\alpha=%s$'%fit3.model.params['smoothing_level'])



ax = df_before.plot(marker='o', color='black', figsize=(12,8))



fcast1.plot(marker='o', ax=ax, color='blue', legend=True)

fit1.fittedvalues.plot(marker='o', ax=ax, color='blue')



fcast2.plot(marker='o', ax=ax, color='red', legend=True)

fit2.fittedvalues.plot(marker='o', ax=ax, color='red')



fcast3.plot(marker='o', ax=ax, color='green', legend=True)

fit3.fittedvalues.plot(marker='o', ax=ax, color='green')



df_air.plot(marker='o', ax=ax, color='orange', legend=True)

plt.show()

# double and triple expoential smoothing is also implemented similarily
# Now for the big one, ARIMA. I am currently reading https://otexts.com/fpp2 so I will take some time

# and get back to this. Stay tuned


