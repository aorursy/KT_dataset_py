# importing the libraries



import numpy as np 

import pandas as pd 



#

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

import matplotlib.pyplot as plt

import io



df = pd.read_csv('/kaggle/input/nba-players-stats/Seasons_Stats.csv')



# kaggle

# df2 = pd.read_csv("../input/airplane-crashes-since-1908/Airplane_Crashes_and_Fatalities_Since_1908.csv")

df.head()
cols = ['Year', 'Player', 'Pos', 'Age', 'Tm', 'G', '2P', '2P%', '2PA', '3P', '3P%', '3PA', 'PTS', 'DRB', 'ORB', 'TRB', 'BLK', 'BLK%', 'PF', 'STL', 'STL%']

df = pd.read_csv('/kaggle/input/nba-players-stats/Seasons_Stats.csv', usecols = cols)

df.head()
for i in range(6,21):

  df[pd.DataFrame(df.iloc[:,i]).columns[0] + " pG"] = (df.iloc[:,i]/df.iloc[:,5]).round(2)



df.head()
a = df.groupby(['Year'])['PTS pG'].mean().round(2)

a = a.reset_index()

a = a.set_index(['Year'])

a.head()



fig, axs = plt.subplots(figsize=(15,10))

fig.suptitle('Points per Game over the years')

axs.plot(a.index, a.loc[:,'PTS pG'].values)
a = df.groupby(['Year'])['3P pG', '2P pG'].mean().round(2)

a = a.reset_index()

a = a.set_index(['Year'])

a.head()



fig, axs = plt.subplots(figsize=(15,10))

ax2 = axs.twinx()

fig.suptitle('3 vs 2 Pointers per Game over the years')



axs.plot(a.index, a.loc[:,'3P pG'].values, label='3P pG', color='red')

axs.tick_params(axis='y', colors='red')



ax2.plot(a.index, a.loc[:,'2P pG'].values, label='2P pG', color='green')

ax2.tick_params(axis='y', colors='green')



axs.legend(loc='upper left')

ax2.legend()
a = df.groupby(['Year'])['3P%', '2P%'].mean().round(2)

a = a.reset_index()

a = a.set_index(['Year'])

a.head()



fig, axs = plt.subplots(figsize=(15,10))

ax2 = axs.twinx()

fig.suptitle('Efficieny of scoring 2 vs 3 PTS over the years')



axs.plot(a.index, a.loc[:,'3P%'].values, label='3P%', color='red')

axs.tick_params(axis='y', colors='red')



ax2.plot(a.index, a.loc[:,'2P%'].values, label='2P%', color='green')

ax2.tick_params(axis='y', colors='green')



axs.legend(loc='upper left')

ax2.legend()
a = df.groupby(['Year'])['STL pG', 'BLK pG'].mean().round(2)

a = a.reset_index()

a = a.set_index(['Year'])

a.head()



fig, axs = plt.subplots(figsize=(15,10))

ax2 = axs.twinx()

fig.suptitle('Steals per Game vs Blocks per Game over the years')



axs.plot(a.index, a.loc[:,'STL pG'].values, label='STL pG', color='red')

axs.tick_params(axis='y', colors='red')



ax2.plot(a.index, a.loc[:,'BLK pG'].values, label='BLK pG', color='green')

ax2.tick_params(axis='y', colors='green')



axs.legend(loc='upper left')

ax2.legend()
a = df.groupby(['Year'])['STL% pG', 'BLK% pG'].mean().round(2)

a = a.reset_index()

a = a.set_index(['Year'])

a.head()



fig, axs = plt.subplots(figsize=(15,10))

ax2 = axs.twinx()

fig.suptitle('Efficieny of Steals and Blocks over the years')



axs.plot(a.index, a.loc[:,'STL% pG'].values, label='STL% pG', color='red')

axs.tick_params(axis='y', colors='red')



ax2.plot(a.index, a.loc[:,'BLK% pG'].values, label='BLK% pG', color='green')

ax2.tick_params(axis='y', colors='green')



axs.legend(loc='upper left')

ax2.legend()
a = df.groupby(['Year'])['ORB pG', 'DRB pG'].mean().round(2)

a = a.reset_index()

a = a.set_index(['Year'])

a.head()



fig, axs = plt.subplots(figsize=(15,10))

ax2 = axs.twinx()

fig.suptitle('Offensive and Defensive rebounds pG over the years')



axs.plot(a.index, a.loc[:,'ORB pG'].values, label='ORB pG', color='red')

axs.tick_params(axis='y', colors='red')



ax2.plot(a.index, a.loc[:,'DRB pG'].values, label='DRB pG', color='green')

ax2.tick_params(axis='y', colors='green')



axs.legend(loc='upper left')

ax2.legend()
a = df['PTS pG'].quantile(0.99)



logic = (df['PTS pG'] >=a)

dfx = df.loc[logic]



logic = (df['PTS pG'] <a)

dfr = df.loc[logic]



# print(df['BLK pG'].median(), dfx['BLK pG'].median())



dfxy = dfx.groupby('Year')['PTS pG'].mean()

dfry = dfr.groupby('Year')['PTS pG'].mean()



fig, axs = plt.subplots(figsize=(15,10))

fig.suptitle('Comparing top 1 percentile vs rest | PTS per Game |')

ax2 = axs.twinx()



axs.plot(dfxy.index, dfxy.values, label='Top 1P', color='blue')

axs.tick_params(axis='y', colors='blue')

ax2.plot(dfry.index, dfry.values, label='Rest', color='red')

ax2.tick_params(axis='y', colors='red')



axs.legend(loc="upper left")

ax2.legend(loc="upper right")

plt.show()
a = df['2P% pG'].quantile(0.99)



logic = (df['2P% pG'] >=a)

dfx = df.loc[logic]



logic = (df['2P% pG'] <a)

dfr = df.loc[logic]



# print(df['BLK pG'].median(), dfx['BLK pG'].median())



dfxy = dfx.groupby('Year')['2P% pG'].mean()

dfry = dfr.groupby('Year')['2P% pG'].mean()



fig, axs = plt.subplots(figsize=(15,10))

fig.suptitle('Comparing top 1 percentile vs rest | 2 pointer accuracy|')

ax2 = axs.twinx()



axs.plot(dfxy.index, dfxy.values, label='Top 1P', color='blue')

axs.tick_params(axis='y', colors='blue')

ax2.plot(dfry.index, dfry.values, label='Rest', color='red')

ax2.tick_params(axis='y', colors='red')



axs.legend(loc="upper left")

ax2.legend(loc="upper right")

plt.show()
a = df['2P pG'].quantile(0.99)



logic = (df['2P pG'] >=a)

dfx = df.loc[logic]



logic = (df['2P pG'] <a)

dfr = df.loc[logic]



# print(df['BLK pG'].median(), dfx['BLK pG'].median())



dfxy = dfx.groupby('Year')['2P pG'].mean()

dfry = dfr.groupby('Year')['2P pG'].mean()



fig, axs = plt.subplots(figsize=(15,10))

fig.suptitle('Comparing top 1 percentile vs rest | 2 pointers |')

ax2 = axs.twinx()



axs.plot(dfxy.index, dfxy.values, label='Top 1P', color='blue')

axs.tick_params(axis='y', colors='blue')

ax2.plot(dfry.index, dfry.values, label='Rest', color='red')

ax2.tick_params(axis='y', colors='red')



axs.legend(loc="upper left")

ax2.legend(loc="upper right")

plt.show()
a = df['3P% pG'].quantile(0.99)



logic = (df['3P% pG'] >=a)

dfx = df.loc[logic]



logic = (df['3P% pG'] <a)

dfr = df.loc[logic]



# print(df['BLK pG'].median(), dfx['BLK pG'].median())



dfxy = dfx.groupby('Year')['3P% pG'].mean()

dfry = dfr.groupby('Year')['3P% pG'].mean()



fig, axs = plt.subplots(figsize=(15,10))

fig.suptitle('Comparing top 1 percentile vs rest | 3 pointer accuracy|')

ax2 = axs.twinx()



axs.plot(dfxy.index, dfxy.values, label='Top 1P', color='blue')

axs.tick_params(axis='y', colors='blue')

ax2.plot(dfry.index, dfry.values, label='Rest', color='red')

ax2.tick_params(axis='y', colors='red')



axs.legend(loc="upper left")

ax2.legend(loc="upper right")

plt.show()
a = df['3P pG'].quantile(0.99)



logic = (df['3P pG'] >=a)

dfx = df.loc[logic]



logic = (df['3P pG'] <a)

dfr = df.loc[logic]



# print(df['BLK pG'].median(), dfx['BLK pG'].median())



dfxy = dfx.groupby('Year')['3P pG'].mean()

dfry = dfr.groupby('Year')['3P pG'].mean()



fig, axs = plt.subplots(figsize=(15,10))

fig.suptitle('Comparing top 1 percentile vs rest | 3 pointers|')

ax2 = axs.twinx()



axs.plot(dfxy.index, dfxy.values, label='Top 1P', color='blue')

axs.tick_params(axis='y', colors='blue')

ax2.plot(dfry.index, dfry.values, label='Rest', color='red')

ax2.tick_params(axis='y', colors='red')



axs.legend(loc="upper left")

ax2.legend(loc="upper right")

plt.show()
a = df['TRB pG'].quantile(0.99)



logic = (df['TRB pG'] >=a)

dfx = df.loc[logic]



logic = (df['TRB pG'] <a)

dfr = df.loc[logic]



# print(df['BLK pG'].median(), dfx['BLK pG'].median())



dfxy = dfx.groupby('Year')['TRB pG'].mean()

dfry = dfr.groupby('Year')['TRB pG'].mean()



fig, axs = plt.subplots(figsize=(15,10))

fig.suptitle('Comparing top 1 percentile vs rest | TRB |')

ax2 = axs.twinx()



axs.plot(dfxy.index, dfxy.values, label='Top 1P', color='blue')

axs.tick_params(axis='y', colors='blue')

ax2.plot(dfry.index, dfry.values, label='Rest', color='red')

ax2.tick_params(axis='y', colors='red')



axs.legend(loc="upper left")

ax2.legend(loc="upper right")

plt.show()
a = df['ORB pG'].quantile(0.99)



logic = (df['ORB pG'] >=a)

dfx = df.loc[logic]



logic = (df['ORB pG'] <a)

dfr = df.loc[logic]



# print(df['BLK pG'].median(), dfx['BLK pG'].median())



dfxy = dfx.groupby('Year')['ORB pG'].mean()

dfry = dfr.groupby('Year')['ORB pG'].mean()



fig, axs = plt.subplots(figsize=(15,10))

fig.suptitle('Comparing top 1 percentile vs rest | ORB |')

ax2 = axs.twinx()



axs.plot(dfxy.index, dfxy.values, label='Top 1P', color='blue')

axs.tick_params(axis='y', colors='blue')

ax2.plot(dfry.index, dfry.values, label='Rest', color='red')

ax2.tick_params(axis='y', colors='red')



axs.legend(loc="upper left")

ax2.legend(loc="upper right")

plt.show()
a = df['DRB pG'].quantile(0.99)



logic = (df['DRB pG'] >=a)

dfx = df.loc[logic]



logic = (df['DRB pG'] <a)

dfr = df.loc[logic]



# print(df['BLK pG'].median(), dfx['BLK pG'].median())



dfxy = dfx.groupby('Year')['DRB pG'].mean()

dfry = dfr.groupby('Year')['DRB pG'].mean()



fig, axs = plt.subplots(figsize=(15,10))

fig.suptitle('Comparing top 1 percentile vs rest | DRB |')

ax2 = axs.twinx()



axs.plot(dfxy.index, dfxy.values, label='Top 1P', color='blue')

axs.tick_params(axis='y', colors='blue')

ax2.plot(dfry.index, dfry.values, label='Rest', color='red')

ax2.tick_params(axis='y', colors='red')



axs.legend(loc="upper left")

ax2.legend(loc="upper right")

plt.show()
a = df['STL pG'].quantile(0.99)



logic = (df['STL pG'] >=a)

dfx = df.loc[logic]



logic = (df['STL pG'] <a)

dfr = df.loc[logic]



# print(df['BLK pG'].median(), dfx['BLK pG'].median())



dfxy = dfx.groupby('Year')['STL pG'].mean()

dfry = dfr.groupby('Year')['STL pG'].mean()



fig, axs = plt.subplots(figsize=(15,10))

fig.suptitle('Comparing top 1 percentile vs rest | STL |')

ax2 = axs.twinx()



axs.plot(dfxy.index, dfxy.values, label='Top 1P', color='blue')

axs.tick_params(axis='y', colors='blue')

ax2.plot(dfry.index, dfry.values, label='Rest', color='red')

ax2.tick_params(axis='y', colors='red')



axs.legend(loc="upper left")

ax2.legend(loc="upper right")

plt.show()
a = df['BLK pG'].quantile(0.99)



logic = (df['BLK pG'] >=a)

dfx = df.loc[logic]



logic = (df['BLK pG'] <a)

dfr = df.loc[logic]



# print(df['BLK pG'].median(), dfx['BLK pG'].median())



dfxy = dfx.groupby('Year')['BLK pG'].mean()

dfry = dfr.groupby('Year')['BLK pG'].mean()



fig, axs = plt.subplots(figsize=(15,10))

fig.suptitle('Comparing top 1 percentile vs rest | BLK |')

ax2 = axs.twinx()



axs.plot(dfxy.index, dfxy.values, label='Top 1P', color='blue')

axs.tick_params(axis='y', colors='blue')

ax2.plot(dfry.index, dfry.values, label='Rest', color='red')

ax2.tick_params(axis='y', colors='red')



axs.legend(loc="upper left")

ax2.legend(loc="upper right")

plt.show()