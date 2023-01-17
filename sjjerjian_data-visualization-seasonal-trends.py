# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns
import matplotlib.pyplot as plt
% matplotlib inline

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
colnames = ['Month','Hire Count','Mean Hire Time']
df = pd.read_excel('../input/tfl-daily-cycle-hires.xls',sheet_name=1,usecols=[3,4,10],names=colnames)

df.dropna(inplace=True) 
df.drop(index=df.index[0],inplace=True)

df['year'],df['month'] = df['Month'].dt.year, df['Month'].dt.month
df.head()
df[['Hire Count','Mean Hire Time']].describe()
print(df.loc[df['Hire Count'].idxmax()])
print('')
print('Total Hires: ' + str(np.sum(df['Hire Count'])))
sns.set_style('white')
f, ax1 = plt.subplots(1,1, figsize=(14, 5), sharex=True,sharey=True)
sns.lineplot(x='Month',y='Hire Count', data=df, ax = ax1)
ax2 = plt.twinx()
sns.lineplot(x='Month',y='Mean Hire Time', data=df, color = 'r', ax = ax2)
ax1.set_xlabel('Year')

df[['Hire Count','Mean Hire Time']].corr()
# 'blip' in 2011 was in April
print(df.loc[df['Mean Hire Time'].idxmax()]) 
def month_to_season(m):
    """convert month to season"""
    if m >= 3 and m < 6:   # March - May
        season = 'Spring'
    elif m >= 6 and m < 9: # June - August
        season = 'Summer'
    elif m >=9 and m < 12:
        season = 'Autumn'  # September - November
    else:
        season = 'Winter'  # December - February
    return season

df['Season'] = df['Month'].dt.month.apply(month_to_season)

scatplot = sns.scatterplot(x='Mean Hire Time',y='Hire Count',data=df,
                           hue='Season',palette='coolwarm',hue_order=['Winter','Autumn','Spring','Summer'])
scatplot.legend(loc=4)
df.groupby(by='Season').mean()
df[(df['Season']=='Winter')].groupby(by='year').mean().drop('month',axis=1)
df[(df['Season']=='Summer')].groupby(by='year').mean().drop('month',axis=1)