import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
delhi=pd.read_csv("../input/delhi-weather-data/testset.csv")

delhi.head()
delhi.drop([' _precipm',' _pressurem',' _heatindexm',' _hum',' _thunder',' _tornado',' _vism',' _wdird',' _wdire',' _wgustm',' _windchillm',' _wspdm'],axis='columns',inplace=True)

delhi.head()
delhi['datetime_utc']=pd.to_datetime(delhi['datetime_utc'])

delhi['datetime_utc'].head()
delhi.isnull().sum()
delhi.dropna(0,inplace=True)

delhi.isnull().sum()
delhiplot=delhi[['datetime_utc',' _tempm']].copy()

delhiplot['just_date'] = delhiplot['datetime_utc'].dt.date



delhifinal=delhiplot.drop('datetime_utc',axis=1)







delhifinal.set_index('just_date', inplace= True)



delhifinal.head()

plt.figure(figsize=(25,8))

plt.plot(delhifinal)

plt.title('Time Series')

plt.xlabel('Date')

plt.ylabel('temperature')

plt.show()



from statsmodels.tsa.stattools import adfuller

def test_stationarity(timeseries):

    

    #Determing rolling statistics

    rolmean = timeseries.rolling(12).mean()

    rolstd = timeseries.rolling(12).std()



    #Plot rolling statistics:

    orig = plt.plot(timeseries, color='blue',label='Original')

    mean = plt.plot(rolmean, color='red', label='Rolling Mean')

    std = plt.plot(rolstd, color='black', label = 'Rolling Std')

    plt.legend(loc='best')

    plt.title('Rolling Mean & Standard Deviation')

    plt.show(block=False)

    

    #Perform Dickey-Fuller test:

    print ('Results of :')

    dftest = adfuller(timeseries, autolag='AIC')

    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])

    for key,value in dftest[4].items():

        dfoutput['Critical Value (%s)'%key] = value

    print(dfoutput)
plt.figure(figsize=(40,8))

test_stationarity(delhifinal)

cond=delhi[' _conds'].unique()

for values in cond:

    print(values)

smoke=delhi[' _conds'][delhi[' _conds'] == 'Smoke'].count()

print(smoke)



rain=delhi[' _conds'][delhi[' _conds'] == 'Rain'].count()

print(rain)

clear=delhi[' _conds'][delhi[' _conds'] == 'Clear'].count()

print(clear)
fog=delhi[' _conds'][delhi[' _conds'] == 'Fog'].count()

print(fog)


Mist=delhi[' _conds'][delhi[' _conds'] == 'Mist'].count()

print(Mist)
Dust=delhi[' _conds'][delhi[' _conds'] == 'Widespread Dust'].count()

print(Dust)
Heavy=delhi[' _conds'][delhi[' _conds'] == 'Heavy Fog'].count()

print(Heavy)
data = {'smoke': smoke,'rain':rain,'clear':clear,'fog':fog,'Mist':Mist,'Dust':Dust,'Heavy':Heavy}

names = list(data.keys())

values = list(data.values())



fig, axs = plt.subplots(1, 3, figsize=(23, 5), sharey=True)

axs[0].bar(names, values,color='Orange')

axs[1].scatter(names, values)

axs[2].plot(names, values, color='Red')

fig.suptitle('Categorical Plotting')
