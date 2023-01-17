#Import modules

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



#Import data and check out the structure

data = pd.read_csv('/kaggle/input/big-five-european-soccer-leagues/BIG FIVE 1995-2019.csv')

data.head()
#Create 2nd half goals columns by subtracting HT from FT

data['H2 Team 1'] = data['FT Team 1'] - data['HT Team 1']

data['H2 Team 2'] = data['FT Team 2'] - data['HT Team 2']



#Rename HT to represent 1/2 halves

data.rename(columns={'HT Team 1':'H1 Team 1','HT Team 2':"H1 Team 2"}, inplace=True)



#Goal difference is given, but is positive for either team that wins - let's create a scale where a negative shows an away win

data['FT OGD'] = data['FT Team 1'] - data['FT Team 2']

data['H1 OGD'] = data['H1 Team 1'] - data['H1 Team 2']

data['H2 OGD'] = data['FT OGD'] - data['H1 OGD']



#Rename year to season, to free up year for calendar year

data.rename(columns={'Year':'Season'}, inplace=True)



#Split the date column into the 5 parts of information

data[['Day of Week', 'Day', 'Month', 'Year', 'Gameweek']] = data['Date'].str.split(' ',expand=True)

data['Day of Week'] = data['Day of Week'].str.strip('()')

data['Gameweek'] = data['Gameweek'].str.strip('()')



#Check out the new first row of data

data.iloc[0]
#Create HomePoints & AwayPoints - datasets that group matches into teams home and away totals for select colummns

HomePoints = data.groupby(['Season','Country','Team 1']).sum().reset_index()[['Season','Country','Team 1','Team 1 (pts)','FT Team 1', 'H1 Team 1', 'H2 Team 1', 

                                                                              'FT Team 2', 'H1 Team 2', 'H2 Team 2']]

AwayPoints = data.groupby(['Season','Country','Team 2']).sum().reset_index()[['Season','Country','Team 2','Team 2 (pts)','FT Team 2', 'H1 Team 2', 'H2 Team 2', 

                                                                              'FT Team 1', 'H1 Team 1', 'H2 Team 1']]



#Create a column that counts the times a team has played

HomeGames = data.groupby(['Season','Country','Team 1']).count()['Team 1 (pts)']

AwayGames = data.groupby(['Season','Country','Team 2']).count()['Team 2 (pts)']



#Add the matches played count to the HomePoints/AwayPoints dataset

HomePoints = HomePoints.reset_index(drop=True)

HomePoints['Matches'] = HomeGames.reset_index()['Team 1 (pts)']

AwayPoints = AwayPoints.reset_index(drop=True)

AwayPoints['Matches'] = AwayGames.reset_index()['Team 2 (pts)']



#Give the dataset columns some better names

HomePoints.rename(columns={'Team 1':'Team', 'Team 1 (pts)':'Points','FT Team 1' : 'GF', 'H1 Team 1' : 'H1 GF', 'H2 Team 1': 'H2 GF',

                           'FT Team 2': 'GA', 'H1 Team 2': 'H1 GA', 'H2 Team 2': 'H2 GA'},inplace=True)

AwayPoints.rename(columns={'Team 2':'Team', 'Team 2 (pts)':'Points','FT Team 2' : 'GF', 'H1 Team 2' : 'H1 GF', 'H2 Team 2': 'H2 GF',

                           'FT Team 1': 'GA', 'H1 Team 1': 'H1 GA', 'H2 Team 1': 'H2 GA'},inplace=True)



#Add a new column to each to tell us that it was home or away before we merge them

HomePoints['Location'] = 'Home'

AwayPoints['Location'] = 'Away'



#Stick AwayPoints to the end of HomePoints and save it to MergePoints

MergePoints = HomePoints.append(AwayPoints)



#Calculate per game metrics and put them in new columns

for col in MergePoints.columns[3:10]:

    colname = str(col) + ' PG'

    MergePoints[colname] = round(MergePoints[col]/MergePoints['Matches'],1)

    

#Check it out

MergePoints.head()
MergePointsSeason = MergePoints.groupby(['Season','Country','Location']).mean().reset_index()

MergePointsSeason.head()
data.describe()
data[data['GGD']==9]
sns.distplot( a=data['Team 1 (pts)'], hist=True, kde=False, rug=False)

plt.xlim(0,3)

plt.show()
sns.distplot( a=data['FT OGD'], bins=range(-9,9), hist=True, kde=False, rug=False, norm_hist=True)

plt.xlim(-9,9)

plt.xlabel('Home Team Goal Difference in a Single Game')

plt.show()
plt.plot(data.groupby('Season')['Team 1 (pts)'].mean())

plt.plot(data.groupby('Season')['Team 2 (pts)'].mean())

plt.title('Home vs Away Points per Game, 1995/96-2019/20')

plt.show()
g = sns.FacetGrid(MergePoints.groupby(['Season','Country','Location']).mean().reset_index(), row="Country", col="Location")

g = g.map(plt.plot, "Points PG")
#In EPL, How have points per game and goals for per game changed over time?

g = sns.FacetGrid(MergePointsSeason[MergePointsSeason['Country'] == 'ENG'], col="Location")

g = g.map(plt.plot, 'Points PG', color='blue')

g = g.map(plt.plot, 'GF PG', color = 'orange')
#In EPL, goals per half

g = sns.FacetGrid(MergePointsSeason[MergePointsSeason['Country'] == 'ENG'], col="Location")

g = g.map(plt.plot, 'H1 GF PG', color='pink')

g = g.map(plt.plot, 'H2 GF PG', color = 'green')
sns.set(rc={'figure.figsize':(14,7)})

ax = sns.boxplot(x="Season", y="Points PG", hue="Location", data=MergePoints[(MergePoints['Country'] == 'ENG') & (MergePoints['Location'] == 'Away')])

plt.title('Distribution of EPL Away Points per Game: 1995-2019')

plt.show()
plt.figure()





plt.subplot(121)

plt.plot(MergePoints[(MergePoints['Country'] == 'ENG') & (MergePoints['Location'] == 'Home')].groupby(['Season'])['Points PG'].quantile(.75),

         label = 'Upper Quartile Home PPG', color='firebrick')

plt.plot(MergePoints[(MergePoints['Country'] == 'ENG') & (MergePoints['Location'] == 'Home')].groupby(['Season'])['Points PG'].quantile(.25),

         color='lightcoral', label = 'Lower Quartile Home PPG')

plt.legend()

plt.ylabel('Points Per Game', fontsize=12)

plt.ylim(0.5, 2.5)





plt.subplot(122)



plt.plot(MergePoints[(MergePoints['Country'] == 'ENG') & (MergePoints['Location'] == 'Away')].groupby(['Season'])['Points PG'].quantile(.75), 

         label='Upper Quartile Away PPG', color='royalblue')

plt.plot(MergePoints[(MergePoints['Country'] == 'ENG') & (MergePoints['Location'] == 'Away')].groupby(['Season'])['Points PG'].quantile(.25),

         label='Lower Quartile Away PPG', color='dodgerblue')

plt.legend()



plt.ylabel('Points Per Game', fontsize=12)

plt.suptitle('Changes in home & away EPL performance, by top & bottom groups of teams', fontsize=16)

plt.ylim(0.5, 2.5)



plt.show()