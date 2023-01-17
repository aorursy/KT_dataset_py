import numpy as np

import pandas as pd
ipl_data = pd.read_excel('../input/DIM_MATCH.xlsx')
ipl_data.head()
#info about the data

ipl_data.info()
#To chechk whether any null values are present

ipl_data.isnull().sum()
#Remove the rows with null values

ipl_data = ipl_data.dropna(axis=0,how='any')

ipl_data.isnull().sum()
ipl_data.shape
ipl_data.columns
#Total number of matches played

ipl_data.Match_SK.count()
#Team with maximium win-margin

ipl_data.loc[ipl_data['Win_Margin'].idxmax()]
#To get the Team

ipl_data.loc[ipl_data['Win_Margin'].idxmax()]['match_winner']
#Team with minimum win-margin

ipl_data.loc[ipl_data['Win_Margin'].idxmin()]['match_winner']
import matplotlib.pyplot as plt

import seaborn as sns

sns.set(style='darkgrid')
#Teams won most number of matches in every ipl season

plt.figure(figsize=(20,10))

sns.countplot(x='Season_Year', data= ipl_data,hue='match_winner',saturation=1)

plt.legend(loc=1)

plt.show()
#maximun number of matches won by team in ipl

sns.countplot(y='match_winner',data=ipl_data)

plt.show()
plt.figure(figsize=(12,8))

sns.countplot(y='Venue_Name',data=ipl_data,palette='rainbow')

plt.tight_layout()
# Highest Man of the matches in ipl

top_player = ipl_data.ManOfMach.value_counts()[:10]
top_player
sns.barplot(x=top_player,y=top_player.index)

plt.show()
#Teams won the match with win margin

ipl_data[ipl_data['Win_Margin']>0].groupby(['match_winner'])['Win_Margin'].apply(np.median).sort_values(ascending=False)
plt.figure(figsize=(12,10))

sns.boxplot(y='match_winner',x='Win_Margin',data=ipl_data[ipl_data['Win_Margin']>0],orient='h')

plt.show()