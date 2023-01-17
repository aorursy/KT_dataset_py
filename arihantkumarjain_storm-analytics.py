# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sbn

import datetime as dt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import matplotlib

import matplotlib.pyplot as plt



%matplotlib inline
storm_df = pd.read_csv("../input/Tornadoes_SPC_1950to2015.csv")
storm_df.head()
storm_df.describe()
storm_df['tz'].value_counts()
storm_df['time'] = pd.to_timedelta(storm_df['time'])
#storm_df.loc[storm_df['tz']==9,'time']

storm_df.loc[storm_df['tz']==9,'time'] = storm_df.loc[storm_df['tz']==9,'time'] - dt.timedelta(hours=5)
#storm_df.loc[storm_df['tz']==0,'date']

storm_df['date'] = pd.to_datetime(storm_df['date'], format='%m/%d/%Y')
print("FC values in the storm data:")

print(storm_df['fc'].value_counts())

print("\n")

print("unknown mag values calculated as per pdf:")

print(storm_df['mag'].value_counts())
storm_df['loss'].value_counts()
storm_df[storm_df['yr']<1996].plot.line('date', 'loss')

storm_df[storm_df['yr']>=1996].plot.line('date', 'loss')
storm_df.loc[(storm_df['yr']>=1996) & (storm_df['loss'] > 0) & (storm_df['loss'] < .00005),'loss'] = 1

storm_df.loc[(storm_df['yr']>=1996) & (storm_df['loss'] >= .00005) & (storm_df['loss'] < .0005),'loss'] = 2

storm_df.loc[(storm_df['yr']>=1996) & (storm_df['loss'] >= .0005) & (storm_df['loss'] < .005),'loss'] = 3

storm_df.loc[(storm_df['yr']>=1996) & (storm_df['loss'] >= .005) & (storm_df['loss'] < .05),'loss'] = 4

storm_df.loc[(storm_df['yr']>=1996) & (storm_df['loss'] >= .05) & (storm_df['loss'] < .5),'loss'] = 5

storm_df.loc[(storm_df['yr']>=1996) & (storm_df['loss'] >= .5) & (storm_df['loss'] < 5),'loss'] = 6

storm_df.loc[(storm_df['yr']>=1996) & (storm_df['loss'] >= 5) & (storm_df['loss'] < 50),'loss'] = 7

storm_df.loc[(storm_df['yr']>=1996) & (storm_df['loss'] >= 50) & (storm_df['loss'] < 500),'loss'] = 8

storm_df.loc[(storm_df['yr']>=1996) & (storm_df['loss'] >= 500) & (storm_df['loss'] <= 5000),'loss'] = 9
storm_df.loc[(storm_df['yr']>=1996) & (storm_df['loss'] >= 500) & (storm_df['loss'] <= 5000),'loss']
storm_df['loss'].value_counts()

    
storm_df['len'] = storm_df['len']*1760
storm_df.plot.scatter('yr','len', figsize=(13,8))

storm_df.plot.scatter('yr','wid', figsize=(13,8))
storm_df[(storm_df['len']>300000) | (storm_df['wid']>3000)]
storm_df.head()
storm_df[['inj','fat','loss','closs','len','wid']].corr()
storm_df.dtypes
names = storm_df.dtypes[(storm_df.dtypes=='int64') | (storm_df.dtypes=='float64')].index

print(names)

tick = np.arange(0,len(names),1)

print(tick)
fig = plt.figure(figsize=(13,13))

ax1 = fig.add_subplot(111)

plt1 = ax1.matshow(storm_df.corr(), vmin=-1, vmax=1, cmap=plt.get_cmap('PuOr'))

fig.colorbar(plt1)

ax1.set_xticks(tick)

ax1.set_xticklabels(names)

ax1.set_yticks(tick)

ax1.set_yticklabels(names)

plt.show()