# loading needed packages



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pylab as plt

from statsmodels.tsa.stattools import adfuller

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

from statsmodels.tsa.arima_model import ARIMA

from scipy.stats import norm

# load the excel

xls = pd.ExcelFile("../input/indicators.xlsx")

# read the inflation sheet

gdp=pd.read_excel(xls,skiprows=1, sheet_name='inflation', na_values='..')

# drop the last two columns on the growth

gdp.drop(['growth 2000-2009', 'growth 2010-2019'], axis=1, inplace=True)

# drop the last 3 rows which does not contain any data

gdp = gdp[:-3]

# melt the data so we have years as column

gdp_melted = pd.melt(gdp, id_vars=['economy'], var_name="Year", value_name="inflation")
# display tail

gdp_melted.tail()
# select the economy

gdp_isdb57 = gdp_melted[gdp_melted['economy']=='IsDB-57']

# create the timeseries

gdpts_isdb57 = pd.Series(gdp_isdb57['inflation'].values,

                        index = pd.DatetimeIndex(data= (tuple(pd.date_range('31/12/1980',

                        periods = 45,

                        freq = 'A-DEC'))),

                                                 freq = 'A-DEC'))
gdpts_isdb57.tail()
# Visualizing Time Series in Python

# Line graph with matplotlib pyplot module

plt.figure(figsize=(12,8))

gdpts_isdb57.plot()

plt.title('Inflation rate for IsDB-57 aggregate')

plt.xlabel('Year')

plt.ylabel('inflation rate (% change)')

plt.legend(['IsDB-57'])
# Test for Stationarity

def stationarity_test(timeseries):

    """"Augmented Dickey-Fuller Test

    Test for Stationarity"""

    from statsmodels.tsa.stattools import adfuller

    print("Results of Dickey-Fuller Test:")

    df_test = adfuller(timeseries, autolag = "AIC")

    df_output = pd.Series(df_test[0:4],

                          index = ["Test Statistic", "p-value", "#Lags Used",

                                   "Number of Observations Used"])

    print(df_output)
stationarity_test(gdpts_isdb57)
# Tests for autocorrelation and partical autocorrelation - Parameters p, q

%matplotlib inline

fig = plt.figure(figsize=(12,8))

ax1 = fig.add_subplot(211)

fig = plot_acf(gdpts_isdb57, lags=20, ax=ax1)

ax2 = fig.add_subplot(212)

fig = plot_pacf(gdpts_isdb57, lags=20, ax=ax2)
# Using ARIMA for the model, with the argument 'order'

# It is easy to change parameters

model = ARIMA(gdpts_isdb57, order=(2, 1, 0))  

results_AR1 = model.fit()

plt.figure(figsize=(12,8))

plt.plot(gdpts_isdb57)

plt.plot(results_AR1.fittedvalues, color='red')
results_AR1.summary()
plt.figure(figsize = (12, 8))

plt.hist(results_AR1.resid, bins = 'auto', density = True, rwidth = 0.85,

         label = 'Residuals') #density TRUE - norm.dist bell curve

mu, std = norm.fit(results_AR1.resid)

xmin, xmax = plt.xlim()

x = np.linspace(xmin, xmax, 100) #linspace returns evenly spaced numbers over a specified interval

p = norm.pdf(x, mu, std) #pdf = probability density function

plt.plot(x, p, 'm', linewidth = 2)

plt.grid(axis='y', alpha = 0.2)

plt.xlabel('Residuals')

plt.ylabel('Density')

plt.title('Residuals 1,0,0 vs Normal Distribution - Mean = '+str(round(mu,2))+', Std = '+str(round(std,2)))

plt.show()
# mean of the residual

np.mean(results_AR1.resid)
model = ARIMA(gdpts_isdb57, order=(2,1,1))  

results_AR211= model.fit()

plt.figure(figsize=(12,8))

plt.plot(gdpts_isdb57)

plt.plot(results_AR211.fittedvalues, color='red')
results_AR211.summary()
model = ARIMA(gdpts_isdb57, order=(2, 2, 1))  

results_AR200 = model.fit()

results_AR200.summary()
# to find the last two terms

gdpts_isdb57.tail(2)
results_AR211.resid.tail()
Fcast211 = results_AR211.predict(start='31/12/2025', end='31/12/2031')
plt.figure(figsize = (12, 8))

plt.plot(gdpts_isdb57, linewidth = 2, label = "original")

plt.plot(Fcast211, color='red', linewidth=2, label='ARIMA 1 0 0')

plt.legend()
Fcast211
# extract uniq economies

economies = gdp_melted.economy.unique()
# need to delete both Palestine and Syria as they do not have sufficient data

economies = np.delete(economies, [39,47])
economies
gdpmc = pd.DataFrame(columns=['economy', 'year', 'gdp'])

#iterate through each economy and calculate forcast

for econ in economies:

    # create the timeseries

    mcts = pd.Series(gdp_melted[gdp_melted['economy']== econ].gdp.values,

                        index = pd.DatetimeIndex(data= (tuple(pd.date_range('31/12/1980',

                        periods = 45,

                        freq = 'A-DEC'))),

                                                 freq = 'A-DEC'))

    mcts.dropna(inplace=True)

    model = ARIMA(mcts, order=(1, 0, 0))  

    results_AR1 = model.fit()

    Fcast100 = results_AR1.predict(start='31/12/2025', end='31/12/2031')

    # now add the forcasted resutls in a dataframe

    year=2025

    for i in Fcast100:

        gdpmc = gdpmc.append({'economy':econ, 'year':year, 'gdp':i}, ignore_index=True)

        year = year+1
# snippet of the resutls 

gdpmc.tail(20)
# unmelting the file to make the years appear as columns

gdpmc_pivot = pd.pivot_table(gdpmc, index=['economy'], columns=['year'])
# export as excel

gdpmc_pivot.to_excel("gdp_forcast.xlsx", sheet_name="GDP Growth")
# export as csv

gdpmc_pivot.to_csv("gdp_forcast.csv")