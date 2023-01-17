import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import os

os.listdir('../input/dream11-ipl2020-live')
data = pd.read_csv('../input/dream11-ipl2020-live/IPL_2020_Daily_Data.csv')

data.head(5)
class data_analytics:

    def __init__(self,data):

        self.data = data

        print('1) Dimension of the dataset is      :',data.shape)

        print('2) Number of Columns in the dataset :',data.shape[1])

        print('3) Number of Rows in the dataset    :',data.shape[0])

        numerical_features = [f for f in data.columns if data[f].dtypes!='O']

        print('4) Count of Numerical Features      :',len(numerical_features))

        cat_features = [c for c in data.columns if data[c].dtypes=='O']

        print('5) Count of Categorical Features    :',len(cat_features))

    def missing_values(self,data):

        print('6) Missing values Estimation        :')

        print('6.1) Total Missing Values in the dataset   :',(data.isnull().sum().sum()))

        print('6.2) Percentage of Total Missing Values    :',(data.isnull().sum().sum()/(data.shape[0]*data.shape[1]))*100)

        print('6.3) Column-wise Missing Values Estimation :')

        for i in data.columns:

            if data[i].isna().sum()>0:

                print(' >> The Column ',i,' has '+ str(data[i].isna().sum()) + ' missing values')
analytics = data_analytics(data)

print(analytics.missing_values(data))
a  = data['Match_number'].max()

print('Total Matches Played by the End of WEEK 1 :',a)
count_of_players = len(data['Player'].unique())

print('Unique Count of Players :',count_of_players)
ipl_teams = len(data['Team'].unique())

ipl_names = data['Team'].unique()

print(f'{ipl_teams} Teams are participating in Dream11 IPL 2020. The Teams are {ipl_names}')
a=data['RH/LH'].value_counts()

rh_players = (a[0]/data['RH/LH'].count())*100

print(f'{round(rh_players,2)} % of the Players are Righted Handed Batsmen.')

lh_players = 100 - rh_players

print(f'{round(lh_players,2)} % of the Players are Left Handed Batsmen.')

print(a)

sns.set(style="darkgrid")

sns.countplot(x="RH/LH", data=data)
data['Ground'].unique()
sns.distplot(data['Runs'])
data['Runs'].describe()
batsmandata = data[['Player','Team','Runs']]

batsmandata.sort_values(by='Runs',ascending=False).head(10)
plt.scatter(data['Runs'],data['RH/LH'])
sns.scatterplot(data=data, x="Role",y="Runs",hue="RH/LH")
batsman_runs = data.groupby(['Player'])['Runs'].sum()

batsman_runs.sort_values(ascending = False, inplace = True)

batsman_runs[:10].plot(x= 'Player', y = 'Runs', kind = 'barh', colormap = 'Accent')
sns.catplot(x="Runs",y="Role",hue="RH/LH",kind="box",data=data)
print(data['Wickets'].value_counts())

data['Wickets'].value_counts().plot(kind='bar',figsize=(5,5))
sns.distplot(data['Wickets'])
bowlerdata = data[['Player','Team','Wickets']]

bowlerdata.sort_values(by='Wickets',ascending=False).head(10)
bowlers_wickets = data.groupby(['Player'])['Wickets'].sum()

bowlers_wickets.sort_values(ascending = False, inplace = True)

bowlers_wickets[:10].plot(x= 'Player', y = 'Wickets', kind = 'barh')
print(data['Dismissal'].value_counts())

sns.countplot(x='Dismissal',data=data)
sns.distplot(data['Dream11_ Points'])
fantasy_points = data.groupby(['Player'])['Dream11_ Points'].sum()

fantasy_points.sort_values(ascending = False, inplace = True)

fantasy_points[:10].plot(x= 'Player', y = 'Dream11_ Points', kind = 'barh', colormap = 'winter_r')
sns.scatterplot(data=data,x="Role",y="Dream11_ Points",hue="RH/LH")
sns.catplot(x="Dream11_ Points",y="Role",hue="RH/LH",kind="box",data=data)
df = pd.read_csv('../input/dream11-ipl2020-live/IPL2020_MatchResults.csv')

analytics = data_analytics(df)

print(analytics.missing_values(df))
df
number_of_matches_stadium = df['GROUND'].value_counts()

print(number_of_matches_stadium)
number_of_wins = df['RESULT (won_by)'].value_counts()

print(number_of_wins)
max_mom = df['MOM'].value_counts().head(1)

print(max_mom)
df['TOSS'].value_counts().head(1)
sum(df['SUPER_OVER']==1)
count = 0

for i in range(df.shape[0]):

    if df['TOSS'][i] == df['RESULT (won_by)'][i]:

        count = count+1

print(f'Out of {df.shape[0]} matches,TOSS Winning teams has managed to WIN only {count}')