# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/testSeries.csv")

df.head()
print(df.shape)

print(df.info())

print(df.isnull().sum())
df.set_index('Team')

# Here we find the teams winning in their Home ground!

df_home_won = df.loc[df.Result == 'Won']

df_home_won.head()
# Let's plot a basic Seaborn Plot!

g = sns.barplot(data = df_home_won , x = 'Team' , y='Won');

plt.xticks(rotation = 60);
# Team Winning in Away Condition:

df_Away_win = df.loc[df.Result == 'Lost']

sns.barplot(data=df_Away_win , x = 'Opponent' , y ='Lost');

plt.xticks(rotation=45);



# Here we see that being Opposition West Indies have had a greater success in Away Conditions
df_home_lost = df.loc[df.Result == 'Lost']

sns.barplot(data=df_Away_win , x = 'Team' , y ='Lost');

plt.xticks(rotation=45);

# Here we see that Australia being in Home have lost more.
sns.lineplot(x="Team", y="Matches",

             data=df)

plt.xticks(rotation=60);

# We see that West indies have performed well over the course till 2019 and is the Best Team of ERA
sns.pairplot(data = df_home_won,hue = 'Opponent')
df_home_won['Date'] = pd.to_datetime(df_home_won['Date'])

df_home_won['Day'] = df_home_won['Date'].dt.day

df_home_won.head()
plt.figure(figsize=(18,9))

df_home_won['Team'] = df_home_won['Team'].astype('category')

sns.boxplot(y = 'Matches' , x = 'Day',data = df_home_won);

plt.show();
#We have changed it to category in order to make it unique!

df_home_won['Team'].unique()
## Let's find the win% for each Team!

df_home_won['Win %'] = round(100*(df_home_won['Won']/df_home_won['Matches']),2)

df_home_won.head()
#Let's plot Teams by Win Percentage!

sns.scatterplot(x = 'Team' , y='Win %',data = df_home_won,hue = 'Drawn');

plt.xticks(rotation = 85);

plt.title('Win % of every team based on the number of Drawn games when played!');
df_home_lost['Win % of Opponent'] = round(100*(df_home_lost['Lost']/df_home_lost['Matches']),2)

df_home_lost.head()
sns.scatterplot(data = df_home_lost,x = 'Opponent',y='Win % of Opponent' , hue = 'Drawn');

plt.xticks(rotation = 85);

plt.xlabel('Win % of every Away Team on number of Drawn games played!');
myTeam_Aus = ['Australia']

df_home_won_Aus = df_home_won[df_home_won.Team.isin(myTeam_Aus)]

Australia_Mean = df_home_won_Aus['Win %'].mean()

myTeam_Eng = ['England']

df_home_won_Eng = df_home_won[df_home_won.Team.isin(myTeam_Eng)]

England_Mean = df_home_won_Eng['Win %'].mean()

myTeam_SA = ['South Africa']

df_home_won_SA = df_home_won[df_home_won.Team.isin(myTeam_SA)]

SA_Mean = df_home_won_SA['Win %'].mean()

myTeam_WI = ['West Indies']

df_home_won_WI = df_home_won[df_home_won.Team.isin(myTeam_WI)]

WI_Mean = df_home_won_WI['Win %'].mean()

myTeam_India = ['India']

df_home_won_India = df_home_won[df_home_won.Team.isin(myTeam_India)]

India_Mean = df_home_won_India['Win %'].mean()

myTeam_NZ = ['New Zealand']

df_home_won_NZ = df_home_won[df_home_won.Team.isin(myTeam_NZ)]

NZ_Mean = df_home_won_NZ['Win %'].mean()

myTeam_PK = ['Pakistan']

df_home_won_PK = df_home_won[df_home_won.Team.isin(myTeam_PK)]

PK_Mean = df_home_won_PK['Win %'].mean()

myTeam_SL = ['Sri Lanka']

df_home_won_SL = df_home_won[df_home_won.Team.isin(myTeam_SL)]

SL_Mean = df_home_won_SL['Win %'].mean()
series_win_mean = pd.Series([Australia_Mean,England_Mean,SA_Mean,WI_Mean,India_Mean,NZ_Mean,PK_Mean,SL_Mean] , index = ['Australia','England','South Africa','West Indies','India'

                                                                                                                       ,'New Zealand','Pakistan','Sri Lanka'])

series_win_mean.sort_values(ascending=False).plot.barh();

plt.ylabel('Teams');

plt.xlabel('Mean Win % of every team!');

plt.title('To find the top 2 teams playing in their Home ground and performing best!!');
myTeam_Aus = ['Australia']

df_home_won_Aus = df_home_lost[df_home_lost.Opponent.isin(myTeam_Aus)]

Australia_Mean = df_home_won_Aus['Win % of Opponent'].mean()

myTeam_Eng = ['England']

df_home_won_Eng = df_home_lost[df_home_lost.Opponent.isin(myTeam_Eng)]

England_Mean = df_home_won_Eng['Win % of Opponent'].mean()

myTeam_SA = ['South Africa']

df_home_won_SA = df_home_lost[df_home_lost.Opponent.isin(myTeam_SA)]

SA_Mean = df_home_won_SA['Win % of Opponent'].mean()

myTeam_WI = ['West Indies']

df_home_won_WI = df_home_lost[df_home_lost.Opponent.isin(myTeam_WI)]

WI_Mean = df_home_won_WI['Win % of Opponent'].mean()

myTeam_India = ['India']

df_home_won_India = df_home_lost[df_home_lost.Opponent.isin(myTeam_India)]

India_Mean = df_home_won_India['Win % of Opponent'].mean()

myTeam_NZ = ['New Zealand']

df_home_won_NZ = df_home_lost[df_home_lost.Opponent.isin(myTeam_NZ)]

NZ_Mean = df_home_won_NZ['Win % of Opponent'].mean()

myTeam_PK = ['Pakistan']

df_home_won_PK = df_home_lost[df_home_lost.Opponent.isin(myTeam_PK)]

PK_Mean = df_home_won_PK['Win % of Opponent'].mean()

myTeam_SL = ['Sri Lanka']

df_home_won_SL = df_home_lost[df_home_lost.Opponent.isin(myTeam_SL)]

SL_Mean = df_home_won_SL['Win % of Opponent'].mean()
series_win_away_mean = pd.Series([Australia_Mean,England_Mean,SA_Mean,WI_Mean,India_Mean,NZ_Mean,PK_Mean,SL_Mean] , index = ['Australia','England','South Africa','West Indies','India'

                                                                                                                       ,'New Zealand','Pakistan','Sri Lanka'])

series_win_away_mean.sort_values(ascending=False).plot.barh();

plt.ylabel('Teams as Opponent!');

plt.xlabel('Mean Win % of every team!');

plt.title('To find the top 2 teams playing in their Away ground and performing best!!');
df_home_won['Year Played'] = df_home_won['Date'].dt.year

df_home_won.head()
#Let's plot the win % for each team based on year Played

plt.figure(figsize=(18,9))

sns.scatterplot(data=df_home_won,x = 'Win %' , y='Year Played',marker = 'o',s=200,hue = 'Team');

plt.title('Teams win % comparison over the years.!');

plt.show();
## Corrleation between the attributes!

cor = df_home_won.corr()

sns.heatmap(cor,annot=True);

plt.show();
cor_away = df_home_lost.corr()

sns.heatmap(cor_away,annot = True);

plt.show();