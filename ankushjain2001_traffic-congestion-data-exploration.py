import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from datetime import datetime, timedelta

import warnings

warnings.filterwarnings("ignore")
data = pd.read_csv('../input/a4020102014/a40_new.csv')

features = list(data.columns)



# Converting str timestamps to pd.Timestamp

data['Date'] = pd.to_datetime(data['Date'],format='%m/%d/%Y')



# Previewing different aspects of the data

#print('Features of the Dataset:\n\n', features)

#print('\nData Type of different Features:\n\n', data.dtypes)

data.head()
numberOfDays = len(data.Date.unique())

print("Total number of Days: ", numberOfDays)

timePeriods = len(data.TimePeriod.unique())

print("Total number of TimePeriods: ", timePeriods)
plt.figure(figsize=(20,5))

plt.plot(data.index, data['Flow'])

plt.title('Traffic Flow')

plt.xlabel('Observation Index')

plt.ylabel('Vehicle Count');

plt.show(block=False)
avgAnnualData = data[['Flow']]

avgAnnualData['Year'] = data['Date'].apply(lambda x: datetime.strftime(x, '%Y'))

avgAnnualData = avgAnnualData.groupby('Year', as_index=False).mean()



plt.figure(figsize=(10,5))

plt.bar(avgAnnualData['Year'], avgAnnualData['Flow'])

plt.title('Average Vehicle Flow for each Year')

plt.xlabel('Year')

plt.ylabel('Average Vehicle Count');

plt.show(block=False)
# Converting the list of TimePeriod into HHMM 24-hr format

timeList = [datetime(2010, 1, 1, 0) + timedelta(minutes=15*x) for x in range(0, timePeriods)]

timeList=[x.strftime('%H%M') for x in timeList]



avgTimePeriodData = data[['TimePeriod', 'Flow']]

avgTimePeriodData = avgTimePeriodData.groupby('TimePeriod', as_index=False).mean()



plt.figure(figsize=(20,5))

plt.bar(avgTimePeriodData['TimePeriod'], avgTimePeriodData['Flow'])

plt.xticks(np.arange(0,96), timeList, rotation='vertical')

plt.title('Average Vehicle Flow for each TimePeriod of the day')

plt.xlabel('TimePeriod')

plt.ylabel('Vehicle Count');

plt.show(block=False)
dayList = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']



weekDayWiseAvgTPData = data[['TimePeriod', 'Flow']]

weekDayWiseAvgTPData['Day'] = data['Date'].apply(lambda x: datetime.strftime(x, '%A'))

weekDayWiseAvgTPData = weekDayWiseAvgTPData.groupby(['Day', 'TimePeriod'], as_index=False).mean()



fig = plt.figure(figsize=(20,20))

fig.tight_layout()

for i in range(7):

    dataWeekly = weekDayWiseAvgTPData[weekDayWiseAvgTPData['Day'] == dayList[i]] # select dataframe with month = i

    ax = fig.add_subplot(6,2,i+1) # add subplot in the i-th position on a grid 12x1  

    ax.title.set_text('Average vehicle flow during the day for '+ dayList[i])

    ax.plot(dataWeekly['TimePeriod'], dataWeekly['Flow'])

    #ax.set_xticks(dataWeekly['TimePeriod'].unique()) # set x axis

    ax.set_xticks([])

    #ax.tick_params(labelrotation=90)
avgDayOfWeekData = data[['Flow']]

avgDayOfWeekData['Day'] = data['Date'].apply(lambda x: x.weekday())#datetime.strftime(x, '%A'))

avgDayOfWeekData = avgDayOfWeekData.groupby(['Day'], as_index=False).mean()

avgDayOfWeekData['Day'] = avgDayOfWeekData['Day'].apply(lambda x: dayList[x])



plt.figure(figsize=(10,5))

plt.plot(avgDayOfWeekData['Day'], avgDayOfWeekData['Flow'])

plt.xticks(np.arange(0,7), dayList, rotation='horizontal')

plt.title('Average Vehicle Flow for each Day of Week')

plt.xlabel('Day')

plt.ylabel('Vehicle Count');

plt.show(block=False)
monthNumList = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']

monthList= ['January','February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']



mthWiseAvgDayOfWeekData = data[['Flow']]

mthWiseAvgDayOfWeekData['Day'] = data['Date'].apply(lambda x: x.weekday())#datetime.strftime(x, '%A'))

mthWiseAvgDayOfWeekData['Month'] = data['Date'].apply(lambda x: datetime.strftime(x, '%m'))

mthWiseAvgDayOfWeekData = mthWiseAvgDayOfWeekData.groupby(['Month','Day'], as_index=False).mean()

mthWiseAvgDayOfWeekData['Day'] = mthWiseAvgDayOfWeekData['Day'].apply(lambda x: dayList[x])



fig = plt.figure(figsize=(20,15))

fig.tight_layout()

for i in range(12):

    dataMonthly = mthWiseAvgDayOfWeekData[mthWiseAvgDayOfWeekData['Month'] == monthNumList[i]] # select dataframe with month = i

    ax = fig.add_subplot(4,3,i+1) # add subplot in the i-th position on a grid 12x1  

    ax.title.set_text('Average vehicle flow during the days for '+ monthList[i])

    ax.plot(dataMonthly['Day'], dataMonthly['Flow'])

    ax.set_xticks(dataMonthly['Day'].unique()) # set x axis

    #ax.set_xticks([])

    #ax.tick_params(labelrotation=45)
monthWiseAvgData = data[['Flow']]

monthWiseAvgData['Month'] = data['Date'].apply(lambda x: datetime.strftime(x, '%m'))

monthWiseAvgData = monthWiseAvgData.groupby(['Month'], as_index=False).mean()



plt.figure(figsize=(10,5))

plt.bar(monthWiseAvgData['Month'], monthWiseAvgData['Flow'])

plt.xticks(np.arange(0,12), monthList, rotation='vertical')

plt.title('Average Vehicle Flow for each Month')

plt.xlabel('Month')

plt.ylabel('Vehicle Count');

plt.show(block=False)
yearList = ['2010','2011','2012','2013','2014']

monthWisePerYearAvgData = data[['Flow']]

monthWisePerYearAvgData['Month'] = data['Date'].apply(lambda x: datetime.strftime(x, '%m'))

monthWisePerYearAvgData['Year'] = data['Date'].apply(lambda x: datetime.strftime(x, '%Y'))

monthWisePerYearAvgData = monthWisePerYearAvgData.groupby(['Year', 'Month'], as_index=False).mean()



fig = plt.figure(figsize=(20,20))

fig.tight_layout()

for i in range(5):

    dataYearly = monthWisePerYearAvgData[monthWisePerYearAvgData['Year'] == yearList[i]] # select dataframe with month = i

    ax = fig.add_subplot(6,2,i+1) # add subplot in the i-th position on a grid 12x1  

    ax.title.set_text('Average vehicle flow during the months for '+ yearList[i])

    ax.plot(dataYearly['Month'], dataYearly['Flow'])

    ax.set_xticks(dataYearly['Month'].unique()) # set x axis

    #ax.set_xticks([])

    #ax.tick_params(labelrotation=90)
'''

# Function to plot the average daily data

#dataTS.reset_index(level=0, inplace=True)

#dataTS['Date'] = dataTS['Date'].apply(lambda x: datetime.strftime(x, '%Y-%m-%d'))

#dataTS.groupby(dataTS.index).mean()

avgDailyData = data[['Date', 'Flow']]

avgDailyData = avgDailyData.groupby('Date').mean()

def plotDailyAvg(avgDailyData):

    plt.figure(figsize=(20,5))

    plt.plot(avgDailyData)

    plt.title('Average Daily Vehicle Flow')

    plt.xlabel('Date')

    plt.ylabel('Vehicle Count');

    plt.show(block=False)

    # data.groupby('Date')['Flow'].mean().plot.line(figsize=(20,5))

'''
# Preparing data for stationarity test

# Combining 'Date' and 'Timeperiod' into a single Timestamp

dataTS_Full = data[['Date', 'Flow']]

dataTS_Full['Date'] = dataTS_Full['Date'].apply(lambda x: datetime.strftime(x, '%d-%m-%Y'))

dataTS_Full['Date'] = dataTS_Full['Date'] + ' ' + data['TimePeriod']

dataTS_Full['Date'] = dataTS_Full['Date'].apply(lambda x: datetime.strptime(x, '%d-%m-%Y %H:%M:%S'))

dataTS_Full = dataTS_Full.rename(columns={'Date':'Timestamp'})

dataTS_Full = dataTS_Full.set_index('Timestamp')



# Converting the dataTS_Full df to series for Dickey-Fuller Test

dataTS_Full = dataTS_Full['Flow']

# Extracting the 3 month subset

dataTS = dataTS_Full['2010-01':'2010-03']

dataTS.describe()
# Importing Dicky Fuller Test

from statsmodels.tsa.stattools import adfuller



# Defining the stationarity test function (Referred from Analytics Vidhya)

def stationarityTest(ts):

    # Calculating the Moving Mean or Moving Average

    rolMean = ts.rolling(96).mean()

    # Calculating the Moving Standard Deviation (Variance = SD^2)

    rolStd = ts.rolling(96).std()



    # Plotting rolMean and rolStd with the TimeSeries

    plt.figure(figsize=(20,5))

    plt.plot(ts, label='Timeseries')

    plt.plot(rolMean, color='red', label='Rolling Mean')

    plt.plot(rolStd, color='green', label='Rolling SD')

    plt.legend(loc='upper left')

    plt.title('Rolling Mean & Standard Deviation')



    # Performing the Dickey-Fuller test

    print('Results of Augmented Dickey-Fuller Test:')

    # The adfuller function takes data in series dtype

    dfTest = adfuller(ts, autolag='AIC')

    dfOutput = pd.Series(dfTest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])

    for key,value in dfTest[4].items():

        dfOutput['Critical Value / Significance Level (%s)'%key] = value

    print(dfOutput)

    

# Performing the test on data

stationarityTest(dataTS)
# Importing decomposition function from statsmodels

from statsmodels.tsa.seasonal import seasonal_decompose

# Performing decomposition for a frequency of 1 week.

decomposition = seasonal_decompose(dataTS, freq=96*7)



trend = decomposition.trend

seasonal = decomposition.seasonal

residual = decomposition.resid



plt.figure(figsize=(20,10))

plt.subplot(411)

plt.plot(dataTS, label='Original')

plt.legend(loc='upper left')

plt.subplot(412)

plt.plot(trend, label='Trend')

plt.legend(loc='upper left')

plt.subplot(413)

plt.plot(seasonal,label='Seasonality')

plt.legend(loc='upper left')

plt.subplot(414)

plt.plot(residual, label='Residuals')

plt.legend(loc='upper left')

plt.tight_layout()
# Checking stationarity of the weekly frequency timeseries by decomposition

residual.dropna(inplace=True)

stationarityTest(residual)
from statsmodels.tsa.stattools import acf, pacf

from statsmodels.graphics.tsaplots import plot_acf

from statsmodels.graphics.tsaplots import plot_pacf



def acf_pacf_plot(ts, lags):

    lagACF = acf(ts, nlags=lags)

    lagPACF = pacf(ts, nlags=lags, method='ols')

    

    plt.figure(figsize=(20,5))

    plot_acf(ts, ax=pyplot.gca(), lags=lags)

    plt.plot(lagACF)

    plt.axhline(y=-1.96/np.sqrt(len(ts.dropna())),linestyle='--',color='gray')

    plt.axhline(y=1.96/np.sqrt(len(ts.dropna())),linestyle='--',color='gray')

    

    plt.figure(figsize=(20,5))

    plot_pacf(series, ax=pyplot.gca(), lags=lags, method='ols')

    plt.plot(lagPACF)

    plt.axhline(y=-1.96/np.sqrt(len(ts.dropna())),linestyle='--',color='gray')

    plt.axhline(y=1.96/np.sqrt(len(ts.dropna())),linestyle='--',color='gray')

acf_pacf_plot(dataTS_Full, 40)