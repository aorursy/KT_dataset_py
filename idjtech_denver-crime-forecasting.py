"""

Documentation for Denver Crime Forecasting using Time Series Analysis



I recently completed the Udemy course: Python for Time Series Analysis and Forecasting with an intense feeling of excitement. On my path to becoming a data scientist, this moment feels like a major milestone. I’ve never felt so eager to apply what I’ve learned.



This is my first real project and I would welcome efficiency and general improvement suggestions. There is probably a much more elegant way to digitise the data than the method I used, which was cumbersome, but functional.



This project does two things:

•	Examines which crime categories vary seasonally

•	Makes forecasts of future crime rates for selected crime categories



Here’s the breakdown

1.	A list of references I used during construction

2.	Library Imports

3.	DataFrame creation from the CSV data file

4.	Set the index to datetime format

5.	DataFrame cleanup – remove unwanted columns

6.	Digitisation, Step 1 – create a template for translating the categorical entries to numeric

7.	Digitisation, Step 2 – make a time series of the crime category to use as a basis for digitising the individual crime categories

8.	Digitisation, Step 2 – make time series for individual crime categories and transform them into digital

9.	Add the digitised crime categories to the DataFrame

10.	Import Seasonality data CSV file – I requested the temperature data for a Denver zip code from https://www.climate.gov/maps-data/dataset/past-weather-zip-code-data-table and made a CSV

11.	Suspecting burglary is seasonal, I overlaid the two data sets

12.	Resample the crime category data to a weekly basis to match the temperature data

13.	Examine all the crime categories for seasonal variation

14.	Preparing for application of forecasting: test for trends and auto-regression/auto-correlation

15.	Use StatsModel Seasonal Decomposition to break out the seasonal element

16.	Test Larceny against seasonal temperatures (with an 80 point boost to get a good overlap)

17.	Test whether Larceny is truly seasonal:

Remove the seasonal element and test for randomness

(it should be random if the seasonal influence has been correctly identified and removed)

18.	Model the data with ARIMA – use the ACF/PACF plots to select model order

19.	Test against other orders to see if we have the best model

20.	Use the modelled data to make forecasts

21.	Try modelling with STL seasonal decomposition, forecast and test residuals

22.	Try Holts-Winters exponential smoothing, forecast and test residuals



"""
# This Python 3 environment comes with many helpful analytics libraries installed# https://www.kaggle.com/kappa420/districts

# https://colorado.hometownlocator.com/zip-codes/zipcodes,city,denver.cfm

# https://www.climate.gov/maps-data/dataset/past-weather-zip-code-data-table

# https://www.usclimatedata.com/climate/denver/colorado/united-states/usco0105

# http://benalexkeen.com/resampling-time-series-data-with-pandas/

# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.resample.html
!pip install stldecompose
import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import statsmodels as sm

import matplotlib.pylab as plb



from pandas.plotting import register_matplotlib_converters

from scipy.stats import norm

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

from statsmodels.tsa.arima_model import ARIMA

from statsmodels.tsa.stattools import adfuller

from statsmodels.tsa.seasonal import seasonal_decompose

from statsmodels.tsa.holtwinters import ExponentialSmoothing

from stldecompose import decompose

from stldecompose import forecast

from stldecompose.forecast_funcs import (naive,

                                         drift, 

                                         mean, 

                                         seasonal_naive)



pd.plotting.register_matplotlib_converters()



%matplotlib inline
denverCrime_df = pd.read_csv('../input/denvercrime/DenverCrime.csv')
denverCrime_df.FIRST_OCCURRENCE_DATE = pd.to_datetime(denverCrime_df.FIRST_OCCURRENCE_DATE)

denverCrime_df.index = pd.DatetimeIndex(denverCrime_df["FIRST_OCCURRENCE_DATE"])
crimeCats = list(denverCrime_df.OFFENSE_CATEGORY_ID.unique())
crimes3_df = denverCrime_df.drop(columns=['INCIDENT_ID','OFFENSE_ID','OFFENSE_CODE','OFFENSE_CODE_EXTENSION',

                                          'OFFENSE_TYPE_ID','OFFENSE_CATEGORY_ID','FIRST_OCCURRENCE_DATE',

                                          'LAST_OCCURRENCE_DATE','REPORTED_DATE','PRECINCT_ID','DISTRICT_ID',

                                          'IS_CRIME','IS_TRAFFIC'])
# Digitisation step 1 - create template

aocSwap={'all-other-crimes':1, 'larceny':0, 'theft-from-motor-vehicle':0, 'traffic-accident':0,

          'drug-alcohol':0, 'auto-theft':0, 'white-collar-crime':0, 'burglary':0, 'public-disorder':0,

          'aggravated-assault':0, 'other-crimes-against-persons':0, 'robbery':0, 'sexual-assault':0,

          'murder':0, 'arson':0}

larcSwap={'all-other-crimes':0, 'larceny':1, 'theft-from-motor-vehicle':0, 'traffic-accident':0,

          'drug-alcohol':0, 'auto-theft':0, 'white-collar-crime':0, 'burglary':0, 'public-disorder':0,

          'aggravated-assault':0, 'other-crimes-against-persons':0, 'robbery':0, 'sexual-assault':0,

          'murder':0, 'arson':0}

tmvSwap={'all-other-crimes':0, 'larceny':0, 'theft-from-motor-vehicle':1, 'traffic-accident':0,

          'drug-alcohol':0, 'auto-theft':0, 'white-collar-crime':0, 'burglary':0, 'public-disorder':0,

          'aggravated-assault':0, 'other-crimes-against-persons':0, 'robbery':0, 'sexual-assault':0,

          'murder':0, 'arson':0}

taSwap={'all-other-crimes':0, 'larceny':0, 'theft-from-motor-vehicle':0, 'traffic-accident':1,

          'drug-alcohol':0, 'auto-theft':0, 'white-collar-crime':0, 'burglary':0, 'public-disorder':0,

          'aggravated-assault':0, 'other-crimes-against-persons':0, 'robbery':0, 'sexual-assault':0,

          'murder':0, 'arson':0}

daSwap={'all-other-crimes':0, 'larceny':0, 'theft-from-motor-vehicle':0, 'traffic-accident':0,

          'drug-alcohol':1, 'auto-theft':0, 'white-collar-crime':0, 'burglary':0, 'public-disorder':0,

          'aggravated-assault':0, 'other-crimes-against-persons':0, 'robbery':0, 'sexual-assault':0,

          'murder':0, 'arson':0}

atSwap={'all-other-crimes':0, 'larceny':0, 'theft-from-motor-vehicle':0, 'traffic-accident':0,

          'drug-alcohol':0, 'auto-theft':1, 'white-collar-crime':0, 'burglary':0, 'public-disorder':0,

          'aggravated-assault':0, 'other-crimes-against-persons':0, 'robbery':0, 'sexual-assault':0,

          'murder':0, 'arson':0}

wccSwap={'all-other-crimes':0, 'larceny':0, 'theft-from-motor-vehicle':0, 'traffic-accident':0,

          'drug-alcohol':0, 'auto-theft':0, 'white-collar-crime':1, 'burglary':0, 'public-disorder':0,

          'aggravated-assault':0, 'other-crimes-against-persons':0, 'robbery':0, 'sexual-assault':0,

          'murder':0, 'arson':0}

burgSwap={'all-other-crimes':0, 'larceny':0, 'theft-from-motor-vehicle':0, 'traffic-accident':0,

          'drug-alcohol':0, 'auto-theft':0, 'white-collar-crime':0, 'burglary':1, 'public-disorder':0,

          'aggravated-assault':0, 'other-crimes-against-persons':0, 'robbery':0, 'sexual-assault':0,

          'murder':0, 'arson':0}

pudiSwap={'all-other-crimes':0, 'larceny':0, 'theft-from-motor-vehicle':0, 'traffic-accident':0,

          'drug-alcohol':0, 'auto-theft':0, 'white-collar-crime':0, 'burglary':0, 'public-disorder':1,

          'aggravated-assault':0, 'other-crimes-against-persons':0, 'robbery':0, 'sexual-assault':0,

          'murder':0, 'arson':0}

assSwap={'all-other-crimes':0, 'larceny':0, 'theft-from-motor-vehicle':0, 'traffic-accident':0,

          'drug-alcohol':0, 'auto-theft':0, 'white-collar-crime':0, 'burglary':0, 'public-disorder':0,

          'aggravated-assault':1, 'other-crimes-against-persons':0, 'robbery':0, 'sexual-assault':0,

          'murder':0, 'arson':0}

ocpSwap={'all-other-crimes':0, 'larceny':0, 'theft-from-motor-vehicle':0, 'traffic-accident':0,

          'drug-alcohol':0, 'auto-theft':0, 'white-collar-crime':0, 'burglary':0, 'public-disorder':0,

          'aggravated-assault':0, 'other-crimes-against-persons':1, 'robbery':0, 'sexual-assault':0,

          'murder':0, 'arson':0}

robSwap={'all-other-crimes':0, 'larceny':0, 'theft-from-motor-vehicle':0, 'traffic-accident':0,

          'drug-alcohol':0, 'auto-theft':0, 'white-collar-crime':0, 'burglary':0, 'public-disorder':0,

          'aggravated-assault':0, 'other-crimes-against-persons':0, 'robbery':1, 'sexual-assault':0,

          'murder':0, 'arson':0}

sexSwap={'all-other-crimes':0, 'larceny':0, 'theft-from-motor-vehicle':0, 'traffic-accident':0,

          'drug-alcohol':0, 'auto-theft':0, 'white-collar-crime':0, 'burglary':0, 'public-disorder':0,

          'aggravated-assault':0, 'other-crimes-against-persons':0, 'robbery':0, 'sexual-assault':1,

          'murder':0, 'arson':0}

murdSwap={'all-other-crimes':0, 'larceny':0, 'theft-from-motor-vehicle':0, 'traffic-accident':0,

          'drug-alcohol':0, 'auto-theft':0, 'white-collar-crime':0, 'burglary':0, 'public-disorder':0,

          'aggravated-assault':0, 'other-crimes-against-persons':0, 'robbery':0, 'sexual-assault':0,

          'murder':1, 'arson':0}

arsSwap={'all-other-crimes':0, 'larceny':0, 'theft-from-motor-vehicle':0, 'traffic-accident':0,

          'drug-alcohol':0, 'auto-theft':0, 'white-collar-crime':0, 'burglary':0, 'public-disorder':0,

          'aggravated-assault':0, 'other-crimes-against-persons':0, 'robbery':0, 'sexual-assault':0,

          'murder':0, 'arson':1}
# Digitisation step 2 - create a basis

denverCrime_ts=pd.Series(denverCrime_df['OFFENSE_CATEGORY_ID'])
# Digitisation step 3 - create individual crime_cat time series and transform into digital

# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.replace.html

aoc_ts = denverCrime_ts.replace(aocSwap)

larceny_ts = denverCrime_ts.replace(larcSwap)

tmv_ts = denverCrime_ts.replace(tmvSwap)

ta_ts = denverCrime_ts.replace(taSwap)

da_ts = denverCrime_ts.replace(daSwap)

at_ts = denverCrime_ts.replace(atSwap)

wcc_ts = denverCrime_ts.replace(wccSwap)

burg_ts = denverCrime_ts.replace(burgSwap)

pudi_ts = denverCrime_ts.replace(pudiSwap)

ass_ts = denverCrime_ts.replace(assSwap)

ocp_ts = denverCrime_ts.replace(ocpSwap)

rob_ts = denverCrime_ts.replace(robSwap)

sex_ts = denverCrime_ts.replace(sexSwap)

murd_ts = denverCrime_ts.replace(murdSwap)

ars_ts = denverCrime_ts.replace(arsSwap)
crimes3_df['AOC'] = aoc_ts

crimes3_df['Larc'] = larceny_ts

crimes3_df['TfMV'] = tmv_ts

crimes3_df['RTA'] = ta_ts

crimes3_df['D&A'] = da_ts

crimes3_df['GTA'] = at_ts

crimes3_df['WCC'] = wcc_ts

crimes3_df['Burg'] = burg_ts

crimes3_df['PuDi'] = pudi_ts

crimes3_df['Ass'] = ass_ts

crimes3_df['OCaP'] = ocp_ts

crimes3_df['Rob'] = rob_ts

crimes3_df['SexAslt'] = sex_ts

crimes3_df['Murd'] = murd_ts

crimes3_df['Ars'] = ars_ts
# Denver seasonal temperatures

# https://www.climate.gov/maps-data/dataset/past-weather-zip-code-data-table

denverWeather_df = pd.read_csv('../input/denverweather/DenverWeather2.csv')

denverWeather_df.DATE = pd.to_datetime(denverWeather_df.DATE)

denverMaxTemps_ts = pd.Series(denverWeather_df['TMAX'].values,

                 index = pd.DatetimeIndex(data = (tuple(pd.date_range('1/1/2014',

                                                                      periods = 2042,

                                                                      freq = 'D'))),

                                          freq = 'D'))
# Is Denver burglary seasonal? 

burgSum = crimes3_df.Burg.resample('W').sum()



rollingTmax = denverMaxTemps_ts.rolling(window=30)

rollingTmax_mean = rollingTmax.mean()



plt.figure(figsize = (15,6))



plt.title('Overall trend of Burglaries in Denver', fontsize=16)

plt.ylabel('Number of Burglaries')

plt.xlabel('Year')

plt.plot(burgSum, label='Burglaries')

rollingTmax_mean.plot(color='green', label='Avg Temp (degF)')

plt.grid(True)

plt.legend()
# Resample the data on a weekly basis to match the temperature data

aocSum = crimes3_df.AOC.resample('W').sum()

larcSum = crimes3_df.Larc.resample('W').sum()

tmvSum = crimes3_df.TfMV.resample('W').sum()

rtaSum = crimes3_df.RTA.resample('W').sum()

daSum = crimes3_df['D&A'].resample('W').sum()

atSum = crimes3_df.GTA.resample('W').sum()

wccSum = crimes3_df.WCC.resample('W').sum()

burgSum = crimes3_df.Burg.resample('W').sum()

pudiSum = crimes3_df.PuDi.resample('W').sum()

assSum = crimes3_df.Ass.resample('W').sum()

ocpSum = crimes3_df.OCaP.resample('W').sum()

robSum = crimes3_df.Rob.resample('W').sum()

sexSum = crimes3_df.SexAslt.resample('W').sum()

murdSum = crimes3_df.Murd.resample('W').sum()

arsSum = crimes3_df.Ars.resample('W').sum()
sumList=[aocSum,larcSum,tmvSum,rtaSum,daSum,atSum,wccSum,burgSum,pudiSum,assSum,ocpSum,robSum,sexSum,murdSum,arsSum]
# let's look at the seasonality of the crime categories

rollingTmax = denverMaxTemps_ts.rolling(window=30)

rollingTmax_mean = rollingTmax.mean()



for i in range(len(sumList)):

    plt.figure(figsize = (15,6))

    plt.title('Overall trend of ' +crimeCats[i]+ ' in Denver', fontsize=16)

    plt.ylabel('Count of ' + crimeCats[i])

    plt.xlabel('Year')

    plt.plot(sumList[i], label=crimeCats[i])

    rollingTmax_mean.plot(color='green', label='Avg Temp (degF)')

    plt.grid(True)

    plt.legend()
# Lets look at larceny...
# Test for Stationarity - does the data show a trend?

def stationarity_test(timeseries):

    """"Augmented Dickey-Fuller Test

    Test for Stationarity"""

    print("Results of Dickey-Fuller Test:")

    df_test = adfuller(timeseries, autolag = "AIC")

    df_output = pd.Series(df_test[0:4],

                          index = ["Test Statistic", "p-value", "#Lags Used",

                                   "Number of Observations Used"])

    print(df_output)
# larceny

denverLarc_ts = pd.Series(crimes3_df.Larc.resample('W').sum(),

                     index = pd.date_range('2014-01-01',

                                           periods = 288,

                                           freq = 'W'))
stationarity_test(denverLarc_ts)
# Dickey-Fuller results imply stationarity/no trend (p-value is < 0.05)
# Is there auto-correlation in raw larceny data?
fig = plt.figure(figsize=(12,8))



ax1 = fig.add_subplot(211)

fig = plot_acf(denverLarc_ts, lags=40, ax=ax1)

ax2 = fig.add_subplot(212)

fig = plot_pacf(denverLarc_ts, lags=40, ax=ax2)
# lots of auto-correlation in the above
# Perform seasonal decompostion and explore...
decompDenverLarc = seasonal_decompose(denverLarc_ts)
dplot = decompDenverLarc.plot()
# is larceny seasonality driven by climate?

denverLarcSeasonAdj = decompDenverLarc.seasonal+80

plt.figure(figsize=(12,8))

denverLarcSeasonAdj.plot(color='blue', label='Denver larceny seasonality')

rollingTmax_mean.plot(color='green', label='Avg Temp (degF)')

plt.grid()

plt.legend()
# but what about when seasonal element removed?

plt.figure(figsize=(12,8))

(decompDenverLarc.observed-decompDenverLarc.seasonal).plot()
# If seasonality is climate-driven (i.e. we have fully identified the dependencies) then the residual should be random
# Histogram of the Residuals

# Importing function for normal distribution

plt.figure(figsize = (12, 8))

plt.hist((decompDenverLarc.observed-decompDenverLarc.seasonal), bins = 'auto', density = True, rwidth = 0.85,

         label = 'De-seasonalised') #density TRUE - norm.dist bell curve

mu, std = norm.fit((decompDenverLarc.observed-decompDenverLarc.seasonal))

xmin, xmax = plt.xlim()

x = np.linspace(xmin, xmax, 100) #linspace returns evenly spaced numbers over a specified interval

p = norm.pdf(x, mu, std) #pdf = probability density function

plt.plot(x, p, 'm', linewidth = 2)

plt.grid(axis='y', alpha = 0.2)

plt.xlabel('Residuals')

plt.ylabel('Density')

plt.title('De-seasoned Larceny vs Normal Distribution - Mean = '+str(round(mu,2))+', Std = '+str(round(std,2)))

plt.show()
# is there auto-correlation with seasonality removed?

fig = plt.figure(figsize=(12,8))



ax1 = fig.add_subplot(211)

fig = plot_acf((decompDenverLarc.observed-decompDenverLarc.seasonal), lags=40, ax=ax1)

ax2 = fig.add_subplot(212)

fig = plot_pacf((decompDenverLarc.observed-decompDenverLarc.seasonal), lags=40, ax=ax2)
# how to make an ARIMA model for Denver burglaries
"""

Here we're looking at p and q arguments for ARIMA: p for the auto-regression, and q for the moving average part.

These two interact quite a bit. So how do we test for auto-regression? It is a visual task with ACF and PACF

plots. The ACF plot excludes the autocorrelation of the shorter lags, the ACF does not.



The plot sheet shows both ACF and PACF plots. It is not always clear how to best start the parameter

selection process from these plots. It of course helps if you know the story behind the data.



Generally the ACF plot tells you the lags for the Moving Average (MA) parameter q, and PACF plot tells

you about the auto-regressive parameter p. However both interact with each other. Once you select one

parameter both the auto-regression and the moving average are affected.



Now in this particular case the ACF plot is outside the threshold, quite a lot. Therefore we should

start with the PACF plot which is significant at lags 7 and 13. We will test them

later on when you have plots like these, you always start with a plot that shows the least amount of

lags outside the benchmark.



It is very important: PACF is the indicator for the auto-regression, p

ACF is the indicator for the moving average part, q

"""
# Using ARIMA for the model, with the argument 'order'

# It is easy to change parameters

model = ARIMA(denverLarc_ts, order=(2, 0, 0))  

results_AR = model.fit()

plt.figure(figsize=(12,8))

plt.grid(True)



plt.plot(denverLarc_ts, label = 'Original data')

plt.plot(results_AR.fittedvalues, color='red', label = 'Model data')
# ARIMA Model Diagnostics

results_AR.summary()
"""

Two important pieces of data to get from the summary are:

AIC - Akaike Information Criteria https://en.wikipedia.org/wiki/Akaike_information_criterion

BIC - Base Informaton Criteria

They are measures of model quality and  - the simpler the better to avoid over-fitting

When comparing models, pick the one with the lowest AIC

"""
# ACF on Residuals of Our Model

fig = plt.figure(figsize=(12,8))

ax1 = fig.add_subplot(211)

fig = plot_acf(results_AR.resid, lags=40, ax=ax1)
# significance still showing at lags 2, 7 and 13 - we can probably improve model therefore

# lags at the front indicate significant need to improve model. Lags at the end may just be coincidence
"""

AN IMPORTANT TEST OF MODEL QUALITY IS THAT THERE SHOULD BE NO PATTERN IN THE RESIDUALS - THEY MUST BE RANDOM

Apply ACF to the residuals to test this

"""
# Histogram of the Residuals

# Importing function for normal distribution

plt.figure(figsize = (12, 8))

plt.hist(results_AR.resid, bins = 'auto', density = True, rwidth = 0.85,

         label = 'Residuals') #density TRUE - norm.dist bell curve

mu, std = norm.fit(results_AR.resid)

xmin, xmax = plt.xlim()

x = np.linspace(xmin, xmax, 100) #linspace returns evenly spaced numbers over a specified interval

p = norm.pdf(x, mu, std) #pdf = probability density function

plt.plot(x, p, 'm', linewidth = 2)

plt.grid(axis='y', alpha = 0.2)

plt.xlabel('Residuals')

plt.ylabel('Density')

plt.title('Residuals 2,0,0 vs Normal Distribution - Mean = '+str(round(mu,2))+', Std = '+str(round(std,2)))

plt.show()
# the above is a little skewed, not a perfect normal distribution, therefore residuals not totally random
# We can readjust the model as often as we like

# Repeat the following procedure for models AR(3), AR(4) and AR(5)

# Which one is the most promising? Look for the lowest AIC
pVar = [3,4,5,6,7]

for var in pVar:

    model = ARIMA(denverLarc_ts, order=(var, 0, 0))  

    results_AR = model.fit()

    print(results_AR.summary())
# AR 6 looks best. Lets examine the residuals in more detail:
model = ARIMA(denverLarc_ts, order=(6, 0, 0))  

results_AR = model.fit()



fig = plt.figure(figsize=(12,8))

ax1 = fig.add_subplot(211)

fig = plot_acf(results_AR.resid, lags=40, ax=ax1)
# still borderline - lags 7 and 13 still fall outside (1 outlier acceptable)
# check the randomness of the residuals again

plt.figure(figsize = (12, 8))

plt.hist(results_AR.resid, bins = 'auto', density = True, rwidth = 0.85,

         label = 'Residuals') #density TRUE - norm.dist bell curve

mu, std = norm.fit(results_AR.resid)

xmin, xmax = plt.xlim()

x = np.linspace(xmin, xmax, 100) #linspace returns evenly spaced numbers over a specified interval

p = norm.pdf(x, mu, std) #pdf = probability density function

plt.plot(x, p, 'm', linewidth = 2)

plt.grid(axis='y', alpha = 0.2)

plt.xlabel('Residuals')

plt.ylabel('Density')

plt.title('Residuals 6,0,0 vs Normal Distribution - Mean = '+str(round(mu,2))+', Std = '+str(round(std,2)))

plt.show()
# not much of an improvement over the 2,0,0. Lets have a look at their forecasts
# Setting up an ARIMA(2,0,0) model and storing its fitted values

model_AR200 = ARIMA(denverLarc_ts, order=(2, 0, 0))  

results_AR200 = model_AR200.fit()
# Forecast based on the ARIMA(2,0,0) model

Fcast200 = results_AR200.predict(start = '2019',

                               end = '2021')

# NOTE: Forecasts have a built-in timestamp
# Setting up an ARIMA(2,0,0) model and storing its fitted values

model_AR600 = ARIMA(denverLarc_ts, order=(6, 0, 0))  

results_AR600 = model_AR600.fit()
Fcast600 = results_AR600.predict(start = '2019',

                               end = '2021')
# Comparing the forecasts via data visualization

plt.figure(figsize = (12, 8))

plt.plot(denverLarc_ts, linewidth = 2, label = "original")

plt.plot(Fcast200, color='red', linewidth = 2,

         label = "ARIMA 2 0 0")

plt.plot(Fcast600, color='blue', linewidth = 2,

         label = "ARIMA 6 0 0")

plt.grid()

plt.legend()
# Not a great performance with ARIMA :(

# Does seasonal decomposition do any better?
# seasonal decomposition with stl package
stl_DenvLarc = decompose(denverLarc_ts, period=52) # 52 because the data has been binned into weeks
stlvisual = stl_DenvLarc.plot()
# check the randomness of the residuals again

plt.figure(figsize = (12, 8))

plt.hist(stl_DenvLarc.resid, bins = 'auto', density = True, rwidth = 0.85,

         label = 'Residuals') #density TRUE - norm.dist bell curve

mu, std = norm.fit(stl_DenvLarc.resid)

xmin, xmax = plt.xlim()

x = np.linspace(xmin, xmax, 100) #linspace returns evenly spaced numbers over a specified interval

p = norm.pdf(x, mu, std) #pdf = probability density function

plt.plot(x, p, 'm', linewidth = 2)

plt.grid(axis='y', alpha = 0.2)

plt.xlabel('Residuals')

plt.ylabel('Density')

plt.title('Residuals STL_larceny vs Normal Distribution - Mean = '+str(round(mu,2))+', Std = '+str(round(std,2)))

plt.show()
fcast_DenvLarc = forecast(stl_DenvLarc, steps=52, fc_func=seasonal_naive, seasonal = True)
fcast_DenvLarc.head()
# Plot of the forecast and the original data

plt.figure(figsize=(12,8))

plt.plot(denverLarc_ts, label='Denver Larceny')

plt.plot(fcast_DenvLarc, label=fcast_DenvLarc.columns[0])

plt.xlim('2014','2021'); plt.ylim(50,240);

plt.grid(True)

plt.legend()
# Perform exponential smoothing

# Setting up the exponential smoothing model (A,N,A) - additive level, no trend, additive seasonality

expsmodel_DenvLarc = ExponentialSmoothing(denverLarc_ts, seasonal = "additive",

                                 seasonal_periods = 52)
# Fit model

expsmodelfit_DenvLarc = expsmodel_DenvLarc.fit()
# Alpha smoothing coefficient

expsmodelfit_DenvLarc.params['smoothing_level']
# Gamma smoothing coefficient

expsmodelfit_DenvLarc.params['smoothing_seasonal']
# coeffs are close or equal to zero. Not surprising as larceny data is fairly smooth - no trends
# Prediction with exponential smoothing

expsfcast_DenvLarc = expsmodelfit_DenvLarc.predict(start = 281, end = 450)
# Plotting the predicted values and the original data

plt.figure(figsize=(12,8))

plt.plot(denverLarc_ts, label='Denver Larceny')

plt.plot(expsfcast_DenvLarc, label='HW forecast')

plt.grid(True)

plt.legend()
# Comparing the model to the original values

# How good is the model fit?

plt.figure(figsize=(12,8))

plt.plot(denverLarc_ts, label='Denver Larceny')

plt.plot(expsmodelfit_DenvLarc.fittedvalues, label='HW model')

plt.grid(True)

plt.xlim('2014','2018'); plt.ylim(70,240);

plt.legend()
# check the randomness of the residuals again

plt.figure(figsize = (12, 8))

plt.hist(expsmodelfit_DenvLarc.resid, bins = 'auto', density = True, rwidth = 0.85,

         label = 'Residuals') #density TRUE - norm.dist bell curve

mu, std = norm.fit(expsmodelfit_DenvLarc.resid)

xmin, xmax = plt.xlim()

x = np.linspace(xmin, xmax, 100) #linspace returns evenly spaced numbers over a specified interval

p = norm.pdf(x, mu, std) #pdf = probability density function

plt.plot(x, p, 'm', linewidth = 2)

plt.grid(axis='y', alpha = 0.2)

plt.xlabel('Residuals')

plt.ylabel('Density')

plt.title('Residuals ExpSmooth Larceny vs Normal Distribution - Mean = '+str(round(mu,2))+', Std = '+str(round(std,2)))

plt.show()