# J. Austin Ellis
# created: 10 March 2018
# update:  10 March 2018

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import date
import os

%matplotlib inline
print(os.listdir("../input"))
cry_df = pd.read_csv('../input/CryptocoinsHistoricalPrices_BTC.csv')
cry2_df = pd.read_csv('../input/CryptocoinsHistoricalPrices.csv')
# Remove redundant column
cry_df.drop(['Unnamed: 0'], axis=1, inplace=True)
cry2_df.drop(['Unnamed: 0'], axis=1, inplace=True)
cry_df.info()
cry_df.describe()
cry_df.columns

cry_df.head()
cry2_df.info()
cry2_df.describe()
cry2_df.columns

cry2_df.head()
cry2_df.dropna(axis=0, inplace=True)
cry2_df['Year'] = cry2_df['Date'].apply(lambda x: x.split('-')[0])
cry2_df['Month'] = cry2_df['Date'].apply(lambda x: x.split('-')[1])
cry2_df['Day'] = cry2_df['Date'].apply(lambda x: x.split('-')[2])
cry2_df.head()
cry2_df = cry2_df.sort_values(['Year', 'Month', 'Day'], ascending=[True, True, True])
cry2_df.head()
startDate = date(2013, 4, 28)
endDate = date(2014, 4, 23)

(endDate - startDate).days
def toDays(d):
    startDate = date(2013, 4, 28)
    dSplit = d.split('-')
    return (date(int(dSplit[0]), int(dSplit[1]), int(dSplit[2])) - startDate).days
def interpTwoDays(x1, y1, x2, y2, step):
    slope = (y2 - y1) / (x2 - x1)
    return slope * step + y1
toDays('2013-07-13')
interpTwoDays(0, 1, 100, 110, 0)
#d = '2013-04-13'
#startDate = date(2013, 4, 28)
#dSplit = d.split('-')
#days = int((date(int(dSplit[0]), dSplit[1], dSplit[2]) - startDate).days)
#type(days)
#startDate = date(2013, 4, 28)


cry2_df['Days'] = cry2_df['Date'].apply(lambda d: toDays(d))

#for dateData in cry2_df.Date:
#    dateArray = dateData.split('-')
#    year = int(dateArray[0])
#    month = int(dateArray[1])
#    day = int(dateArray[2])
#    print((date(year, month, day) - startDate).days)

cry2_df.head(10)
btc_df = cry_df.loc[(cry_df.coin == 'BTC')]
btc2_df = cry2_df.loc[(cry2_df.coin == 'BTC')]

btc2_vol_df = btc2_df.loc[(btc2_df.Volume != '-')]

#btc2_df.tail(100)
btc2_df.head(10)
btc2_df = btc2_df.reset_index(drop=True)
btc2_df.head()

interp_btc2_df = pd.DataFrame(np.arange(btc2_df['Days'].min(), btc2_df['Days'].max()), columns=['Days'])
interp_btc2_df['Close'] = 0.0

interp_btc2_df.head()
dsize = btc2_df['Days'].size

for i in range(dsize):
    if (i == dsize - 1):
        interp_btc2_df['Close'].loc[dsize - 1] = btc2_df['Close'].loc[i]
        break
        
    d1 = btc2_df['Days'].loc[i]
    d2 = btc2_df['Days'].loc[i + 1]
    
    y1 = btc2_df['Close'].loc[i]
    y2 = btc2_df['Close'].loc[i + 1]
    
#    if (i < 10):
#        print('{0:d}; {1:d}; {2:+f}; {3:+f}; ROW: {4:d}'.format(d1,d2,y1,y2, i))
    
    for j in range(d1, d2):
        interp = interpTwoDays(d1, y1, d2, y2, j - d1)
#        if (j < 15):
#            print('{0:d}: {1:+f}'.format(j, interp))
        interp_btc2_df.at[j, 'Close'] = interp
interp_btc2_df.tail(100)

plt.figure(figsize=(20,10))
interp_btc2_df.plot(x='Days', y='Close', kind='area')

dfreq = 1.0; # data freq
dsize = interp_btc2_df['Days'].size; # data points

T = dsize / dfreq
frq = np.arange(dsize) # freq range
Y = np.fft.fft(interp_btc2_df['Close']) / dsize

fig, ax = plt.subplots(3, 1)
ax[0].plot(interp_btc2_df['Days'],interp_btc2_df['Close'])
ax[0].set_xlabel('Days')
ax[0].set_ylabel('Close')
ax[1].plot(frq[range(50)],Y[range(50)],'r') # plotting the spectrum
ax[1].set_xlabel('Freq (1/Days)')
ax[1].set_ylabel('|Y|')
ax[2].plot(frq,Y,'r') # plotting the spectrum
ax[2].set_xlabel('Freq (1/Days)')
ax[2].set_ylabel('|Y|')
plt.plot
