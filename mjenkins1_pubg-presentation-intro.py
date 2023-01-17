# Import libraries 

import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns 

import os

import warnings



%matplotlib inline

warnings.filterwarnings("ignore")
# Set the size of the plots 

plt.rcParams["figure.figsize"] = (18,8)

sns.set(rc={'figure.figsize':(18,8)})
data = pd.read_csv("../input/pubg-finish-placement-prediction/train_V2.csv")

print("Finished loading the data")
data.shape
data.info()
data.head()
data.drop(columns=['rankPoints'], inplace=True)
# Check to see what we are dealing with regarding missing and null values 

data.isnull().values.any()
data.isnull().sum()
data.dropna(inplace=True)

data.isnull().values.any()
# Check to see win percentage distribution 

sns.distplot(data['winPlacePerc']).set_title('Distribution of Winning Percentile');
print('Mean: {:.4f}, Median {:.4f}'.format(data['winPlacePerc'].mean(), data['winPlacePerc'].median()))
data['matchMean'] = data.groupby('matchId')['winPlacePerc'].transform('mean')

data['matchMedian'] = data.groupby('matchId')['winPlacePerc'].transform('median')
sns.distplot(data['matchMean'], kde=False).set_title('Mean for Winning Percentile grouped by match');
sns.distplot(data['matchMedian'], kde=False).set_title('Median for Winning Percentile grouped by match');
# Get values

print('Mean: {:.4f}, Median {:.4f}'.format(data['matchMean'].mean(), data['matchMedian'].median()))
# Can do this with matchType and then derive the team and match size

data['matchType'].unique()
sns.countplot('matchType', data=data);
data['teamSize'] = data.groupby('groupId')['groupId'].transform('count')

data['maxTeamSize'] = data.groupby('matchId')['teamSize'].transform('max')

data['matchSize'] = data.groupby('matchId')['Id'].transform('nunique')
sns.distplot(data['matchSize'], kde=False).set_title('Distribution of Players per Game');
# Let's see the largest team size

data['maxTeamSize'].max()
sns.distplot(data['teamSize'], kde=False);
types = ['solo', 'solo-fpp', 'duo', 'duo-fpp', 'squad', 'squad-fpp']

data = data.loc[data['matchType'].isin(types)]
sns.countplot('matchType', data=data);
sns.distplot(data['matchSize'], kde=False).set_title('Distribution of Players per Game');sns.distplot(data['matchSize'], kde=False).set_title('Distribution of Players per Game');
data['matchSize'].min()
sns.distplot(data['teamSize'], kde=False);
# Also look at top 10% and bottom 10% of players 

top_10 = data[data['winPlacePerc'] >= 0.9]

bottom_10 = data[data['winPlacePerc'] <= 0.1]
data['boosts'].unique()
sns.scatterplot(x="boosts", y="winPlacePerc", data=data, color='seagreen');
sns.scatterplot(x="boosts", y="winPlacePerc", data=top_10, color='seagreen');
sns.scatterplot(x="boosts", y="winPlacePerc", data=bottom_10, color='seagreen');
sns.scatterplot(x="heals", y="winPlacePerc", data=data, color='seagreen');
sns.scatterplot(x="heals", y="winPlacePerc", data=top_10, color='seagreen');
sns.scatterplot(x="heals", y="winPlacePerc", data=bottom_10, color='seagreen');
top_10[['boosts', 'heals']].describe()
bottom_10[['boosts', 'heals']].describe()
# Count 

sns.countplot(data['kills'], color='red');
sns.lineplot(x="kills", y='winPlacePerc', data=data, color='red');
sns.scatterplot(x="kills", y="winPlacePerc", data=data, color='red');
sns.scatterplot(x="kills", y="winPlacePerc", data=top_10, color='red');
sns.scatterplot(x="kills", y="winPlacePerc", data=bottom_10, color='red');
zero_kills = data.copy()

zero_kills = zero_kills[zero_kills['kills']==0]
# Same reason as previous line

sns.scatterplot(x="kills", y='winPlacePerc', data=zero_kills);
sns.lineplot(x="killPlace", y='winPlacePerc', data=zero_kills);
data.head()
data[data['groupId'] == '4d4b580de459be'][['matchType', 'kills', 'killPlace', 'winPlacePerc']]
data[data['matchType'] == 'duo-fpp'].head()
data[data['groupId'] == '8e0a0ea95d3596'][['matchType', 'kills', 'killPlace', 'winPlacePerc']]
sns.scatterplot(x="damageDealt", y="winPlacePerc", data=data);
sns.scatterplot(x="damageDealt", y="winPlacePerc", data=top_10);
sns.scatterplot(x="damageDealt", y="winPlacePerc", data=bottom_10);
sns.scatterplot(x="matchDuration", y="winPlacePerc", data=data, color='yellow');
sns.scatterplot(x="matchDuration", y="winPlacePerc", data=top_10, color='yellow');
sns.scatterplot(x="matchDuration", y="winPlacePerc", data=bottom_10, color='yellow');
sns.scatterplot(x="killPoints", y="winPlacePerc", data=data, color='orange');
sns.scatterplot(x="killPoints", y="winPlacePerc", data=top_10, color='orange');
sns.scatterplot(x="killPoints", y="winPlacePerc", data=bottom_10, color='orange');
sns.lineplot(x="killPoints", y='kills', data=data, color='orange');
sns.lineplot(x="kills", y='killPoints', data=data, color='orange');
sns.lineplot(x="winPoints", y='winPlacePerc', data=data, color='brown');
sns.scatterplot(x="winPoints", y="winPlacePerc", data=data, color='brown');
sns.scatterplot(x="winPoints", y="winPlacePerc", data=top_10, color='brown');
sns.scatterplot(x="winPoints", y="winPlacePerc", data=bottom_10, color='brown');