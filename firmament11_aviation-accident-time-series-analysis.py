%matplotlib inline

import warnings

warnings.simplefilter('ignore')

import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

from scipy import stats

import statsmodels.api as sm
aas = pd.read_csv("../input/AviationData.csv", encoding = 'iso-8859-1')
aas.duplicated().any()

aas.drop_duplicates(inplace=True)
Cols_to_Drop = ['Investigation.Type', 'Accident.Number', 'Location', 'Country', 'Latitude', 'Longitude', 

                'Airport.Code', 'Airport.Name', 'Aircraft.Category', 'Registration.Number',

                'Engine.Type', 'FAR.Description', 'Schedule', 'Purpose.of.Flight',

                'Air.Carrier', 'Broad.Phase.of.Flight', 'Report.Status', 'Publication.Date']

aas.drop(columns = Cols_to_Drop, inplace = True)
aas['Total.Fatal.Injuries'].fillna(0, inplace=True)

aas['Total.Serious.Injuries'].fillna(0, inplace=True)

aas['Total.Minor.Injuries'].fillna(0, inplace=True)

aas['Total.Uninjured'].fillna(0, inplace=True)

aas['Aircraft.Damage'].fillna('Unknown', inplace=True)

aas['Make'].fillna('Unknown', inplace=True)

aas['Model'].fillna('Unknown', inplace=True)

aas['Weather.Condition'].fillna('UNK', inplace=True)
grouped_make = aas.groupby('Make')['Amateur.Built']

aas['Amateur.Built'] = aas['Amateur.Built'].fillna(grouped_make.transform('first'))





#There are only 29 null values left, and never heard of the companies so I fill it with Yes

aas['Amateur.Built'].isnull().value_counts()

aas['Amateur.Built'].fillna('Yes', inplace=True)
#Ballon has no engine so setting it equal to 0

aas['Number.of.Engines'] = aas['Number.of.Engines'].mask(aas['Number.of.Engines'].isnull() &

                                                         aas['Make'].str.contains('balloon', case=False), 0)





#To see which Company has the ability to build more than 2 engines

aas[aas['Number.of.Engines'] > 2]['Make'].value_counts()



#Boeing 747 and 707 models have 4 engines, and 727 has 3 engines

aas['Number.of.Engines'] = aas['Number.of.Engines'].mask(aas['Number.of.Engines'].isnull() &

                                                         aas['Make'].str.contains('boeing', case=False) &

                                                         aas['Model'].str.contains('747', case=False), 4)

aas['Number.of.Engines'] = aas['Number.of.Engines'].mask(aas['Number.of.Engines'].isnull() &

                                                         aas['Make'].str.contains('boeing', case=False) &

                                                         aas['Model'].str.contains('707', case=False), 4)

aas['Number.of.Engines'] = aas['Number.of.Engines'].mask(aas['Number.of.Engines'].isnull() &

                                                         aas['Make'].str.contains('boeing', case=False) &

                                                         aas['Model'].str.contains('727', case=False), 3)





#MCDONNELL DOUGLAS DC and DH models have 3 and 4 engines respectively

aas['Number.of.Engines'] = aas['Number.of.Engines'].mask(aas['Number.of.Engines'].isnull() &

                                                         aas['Make'].str.contains('douglas', case=False) &

                                                         aas['Model'].str.contains('DC', case=False), 3)

aas['Number.of.Engines'] = aas['Number.of.Engines'].mask(aas['Number.of.Engines'].isnull() &

                                                         aas['Make'].str.contains('douglas', case=False) &

                                                         aas['Model'].str.contains('DH', case=False), 4)





#For the rest of the companies, assume that they dont have the ability to build more than 2 engines (either 1 or 2)

#Based on the models, series starting under 3 have one engine and more than 3 have two engines

aas['ModelExtract'] = aas['Model'].str.extract('(\d)', expand=True)

aas['ModelExtract'].fillna(0, inplace=True)

aas['ModelExtract'] = pd.to_numeric(aas['ModelExtract'])

aas['Number.of.Engines'] = aas['Number.of.Engines'].mask(aas['Number.of.Engines'].isnull() &

                                                         aas['ModelExtract'] < 3, 1)

aas['Number.of.Engines'] = aas['Number.of.Engines'].mask(aas['Number.of.Engines'].isnull() &

                                                         aas['ModelExtract'] >= 3, 2)







aas['Event.Date'] = pd.to_datetime(aas['Event.Date'])

aas['Year'] = aas['Event.Date'].dt.year

#or aas['Year'] = aas['Year'].apply(lambda x: x.year)

aas['Month'] = aas['Event.Date'].dt.month

aas = aas[aas['Year'] >= 1982]
aas['Injuries'] = aas['Total.Fatal.Injuries'] + aas['Total.Serious.Injuries'] + aas['Total.Minor.Injuries']
aas.head()
accidents_per_year = aas.groupby('Year').size()

fig = plt.figure()

subplot = fig.add_subplot(1,1,1)

accidents_per_year.plot(ax=subplot, marker = 'o')

subplot.set_xlabel('Accident Year')

subplot.set_ylabel('# of Accidents')

subplot.set_title('Accident per Year')
apy_return = np.log(accidents_per_year)

apy_return = apy_return[1:]
fig = plt.figure(figsize=(13,8))

ax2 = fig.add_subplot(212)



fig = sm.graphics.tsa.plot_acf(apy_return.values, lags=10, ax=ax2)
fig = plt.figure(figsize=(13,8))

ax2 = fig.add_subplot(212)



fig = sm.graphics.tsa.plot_pacf(apy_return.values, lags=10, ax=ax2)
arima_model = sm.tsa.ARIMA(apy_return, (2, 0, 4)).fit(disp=False)
sm.stats.durbin_watson(arima_model.resid)
stats.normaltest(arima_model.resid)
figure = plt.figure(figsize=(16, 8))



plt.plot(apy_return, color = 'red', alpha=0.5, label="Temperatures")

plt.plot(arima_model.fittedvalues, color='green', label="Fitted")