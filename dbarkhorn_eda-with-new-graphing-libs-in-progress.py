import pandas as pd

import missingno as msno

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

from statsmodels.tsa.arima_model import ARIMA

from statsmodels.tsa.stattools import adfuller
%matplotlib inline
master_data = pd.read_csv('../input/Crime_Data_2010_2017.csv')
master_data.columns
sorted_data = msno.nullity_sort(master_data, sort='descending')

msno.matrix(sorted_data)
master_data['Date Occurred'] = pd.to_datetime(master_data['Date Occurred'])

master_data['Time master_datarred'] = pd.to_datetime(master_data['Time Occurred'])
cols = master_data.columns[0:-1]

cols = cols.insert(len(cols), 'Location')

master_data.columns = cols
Location_X = []

Location_Y = []



for i in master_data['Location']:

    if(isinstance(i, str)):

        locs = i[1:-1].split(',')

        Location_X.append(float(locs[0]))

        Location_Y.append(float(locs[1][1:]))

    else:

        Location_X.append(0)

        Location_Y.append(0)



master_data['Location_X'] = Location_X

master_data['Location_Y'] = Location_Y
loc_data = master_data.loc[master_data['Location_X'] != 0]
msno.geoplot(loc_data.sample(100000), x='Location_X', y='Location_Y', histogram="True")
master_data['Crime Code Description'].unique()
plt.figure(figsize=(10,30))

count = sns.countplot(

    data = master_data, 

    y = 'Crime Code Description',

)

plt.show()
vehicle_codes = ['VEHICLE - STOLEN', 

                 'BURGLARY FROM VEHICLE', 

                 'THEFT FROM MOTOR VEHICLE - ATTEMPT', 

                 'BURGLARY FROM VEHICLE, ATTEMPTED',

                 'VEHICLE - ATTEMPT STOLEN',

                 'THEFT FROM MOTOR VEHICLE - PETTY ($950 & UNDER)',

                 'THEFT FROM MOTOR VEHICLE - GRAND ($400 AND OVER)'

                 

                ]

vehicle_data = master_data.loc[master_data['Crime Code Description'].isin(vehicle_codes)]

vehicle_data_loc = vehicle_data.loc[master_data['Location_X'] != 0]
sorted_data_V = msno.nullity_sort(vehicle_data_loc, sort='descending')

msno.matrix(sorted_data_V)
vehicle_data = vehicle_data.drop(

    labels=['Weapon Used Code', 

            'Weapon Description', 

            'Crime Code 2', 

            'Crime Code 3', 

            'Crime Code 4', 

            'Cross Street'],

    axis=1)
vehicle_data['Month Occurred'] = vehicle_data['Date Occurred'].dt.month

vehicle_data['Day of Week Occurred'] = vehicle_data['Date Occurred'].dt.dayofweek

vehicle_data['Year Occurred'] = vehicle_data['Date Occurred'].dt.year

vehicle_data['Month and Year Occurred'] = list(zip(vehicle_data['Year Occurred'], vehicle_data['Month Occurred']))
plt.figure(figsize=(15,10))

sns.countplot(x = 'Month Occurred', data = vehicle_data)

plt.show()
plt.figure(figsize=(15,10))

sns.countplot(x = 'Day of Week Occurred', data = vehicle_data)

plt.show()
vehicle_data_TSM = vehicle_data.groupby('Month and Year Occurred').count()['DR Number']
# ignoring last value as it is an outlier

plt.figure(figsize=(15,10))

plt.plot(vehicle_data_TSM.values[:-1])

plt.xticks([i * 12 for i in range(9)], ('2010', '2011', '2012', '2013', '2014', '2015', '2016', '2017'))

plt.xlabel('Date')

plt.ylabel('Number of vehicle related crimes')

plt.title('Number of Vehicle Related Crimes in Los Angeles (grouped by Month)')

plt.show()
vehicle_data_TSD = vehicle_data.groupby('Date Occurred').count()['DR Number']
# Removed last 30 days as they may be outliers-- ie not yet reported

plt.figure(figsize=(15,10))

plt.plot(vehicle_data_TSD.values[:-30])

plt.xticks([i * 365 for i in range(9)], ('2010', '2011', '2012', '2013', '2014', '2015', '2016', '2017'))

plt.xlabel('Date')

plt.ylabel('Number of vehicle related crimes')

plt.title('Number of Vehicle Related Crimes in Los Angeles (grouped by Day)')

plt.show()
# Testing and training split

trainV_TSM = vehicle_data_TSM[:int(.8 * len(vehicle_data_TSM))]

testV_TSM = vehicle_data_TSM[int(.8 * len(vehicle_data_TSM)):]

trainV_TSD = vehicle_data_TSD[:int(.8 * len(vehicle_data_TSD))]

trainV_TSD = vehicle_data_TSD[int(.8 * len(vehicle_data_TSD)):]
plot_acf(trainV_TSM)

plt.show()
plot_acf(trainV_TSD, lags=np.arange(80))

plt.show()
plot_acf(trainV_TSD, lags=np.arange(10))

plt.show()
plot_pacf(trainV_TSD, lags=np.arange(30))

plt.show()
adf = adfuller(trainV_TSD.diff(1)[1:])

print('ADF Statistic: %f' % adf[0])

print('p-value: %f' % adf[1])

print('Critical Values', adf[4]['1%'])
plt.plot(trainV_TSD.diff(1)[1:])

plt.title('Daily differenced TS crime data')

plt.show()
# fit model, converting to float to address bug found here https://github.com/statsmodels/statsmodels/issues/3504

trainV_TSD = trainV_TSD.astype(float)

model = ARIMA(trainV_TSD, order=(7,1,0))

model.fit(method='css')
#cool graph

sns.jointplot(

    x='Location_X', 

    y='Location_Y', 

    data=vehicle_data_loc, 

    size=15, 

    kind='kde')