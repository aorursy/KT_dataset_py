# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from scipy import stats





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

data = pd.read_csv("../input/data.csv")

df= pd.DataFrame(data)

# Any results you write to the current directory are saved as output.
print(df.head())
pd.set_option('display.max_column',10)

pd.set_option('display.max_rows',10)

sns.set(style = 'dark', palette = 'colorblind', color_codes = True)
print(data.info)
print(data.columns)
print(data.describe())
print(df.shape)
plt.rcParams['figure.figsize'] = (20,16)

HM = sns.heatmap(df[['Age', 'Overall', 'Potential', 'Value', 'Wage',

                'Acceleration', 'Aggression', 'Agility', 'Balance', 'BallControl', 

                'Body Type','Composure', 'Crossing','Dribbling', 'FKAccuracy', 'Finishing', 

                'HeadingAccuracy', 'Interceptions','International Reputation',

                'Joined', 'Jumping', 'LongPassing', 'LongShots',

                'Marking', 'Penalties', 'Position', 'Positioning',

                'ShortPassing', 'ShotPower', 'Skill Moves', 'SlidingTackle',

                'SprintSpeed', 'Stamina', 'StandingTackle', 'Strength', 'Vision',

                'Volleys']].corr(),annot = True, cmap = 'Blues', linewidths = 0.5)

HM.set_title(label = 'Heatmap of the Dataset', fontsize = 20)

print(HM)

plt.show()
quality = ('Agility', 'Balance', 'Dribbling', 'SprintSpeed')

for index, qual in enumerate(quality):

   plt.subplot(len(quality)/4+1, 4, index+1)

   sns.regplot(x = 'Acceleration', y = qual, data = df)

plt.show()
print('ELDEST PLAYERS')

eldest = df.sort_values('Age', ascending = False)[['Name','Nationality','Age']].head(10)

print(eldest)
print('YOUNGEST PLAYERS')

youngest = df.sort_values('Age', ascending = True)[['Name','Nationality','Age']].head(10)

print(youngest)
some_clubs = ('Juventus', 'Real Madrid', 'Paris Saint-Germain', 'FC Barcelona', 'Legia Warszawa', 'Manchester United')

df_club = df.loc[df['Club'].isin(some_clubs) & df['Age']]

print(df_club.head(5))

fig, CAax = plt.subplots()

fig.set_size_inches(10,20)

CAax = sns.violinplot(x = 'Club', y = 'Age' , data = df_club)

CAax.set_title('Distribution of Age in some clubs', fontsize = 20)

plt.show()
top_30 = df['Nationality'].value_counts()[:30]

plt.figure(figsize=(16,10))

sns.barplot(top_30.index, top_30.values)

plt.xticks(rotation=45)

plt.title('Most frequent nationality of player')

plt.show()
plt.figure(figsize=(16,10))

sns.countplot(x='Preferred Foot', data=df)

plt.title('Left Foot Vs Right Foot')
#Oldest Team

Oldest = df.groupby('Club')['Age'].mean().sort_values(ascending = False).head()

print(Oldest.head(5))
#Youngest Team

Youngest = df.groupby('Club')['Age'].mean().sort_values(ascending = True).head()

print(Youngest.head(5))
some_club = ['Manchester United', 'Manchester City', 'FC Barcelona', 'Real Madrid','Chelsea','Juventus']

df_clubs = df.loc[df['Club'].isin(some_club) & df['Age'] & df['Overall']]

print(df_clubs.head())
fig, Clax = plt.subplots()

fig.set_size_inches(10,10)

Clax = sns.barplot(x = df_clubs['Club'], y = df_clubs['Overall'], palette = 'rocket' )

Clax.set_title('Relation between Club and their Overalls', fontsize = 16)

plt.show()
#Top 10 Left Footed Players

print(df[df['Preferred Foot'] == 'Left'][['Name','Club','Overall']].head(10))
#Top 10 Right Footed Players

print(df[df['Preferred Foot'] == 'Right'][['Name','Club','Overall']].head(10))
sns.lmplot(x = 'BallControl', y = 'Dribbling', data = df, scatter_kws = {'alpha':1} ,col = 'Preferred Foot')

plt.show()
sns.jointplot(x = 'Dribbling', y = 'Crossing', data = df, kind = 'hex')
def Value_to_int(df_value):

    try:

        value = float(df_value[1:-1])

        suffix = df_value[-1:]

        if suffix == 'M':

            value = value * 1000000

        elif suffix == 'K':

            value = value * 1000

    except ValueError:

        value = 0

    return value

df['Value_float'] = df['Value'].apply(Value_to_int)

print('Top 10 Clubs with Market Value')

print(df.groupby('Club')['Value_float'].sum().sort_values(ascending = False).head(10))
def Wage_to_float(df_wage):

    try:

        wage = float(df_wage[1:-1])

        suffix = df_wage[-1:]

        if suffix == 'K':

            wage = wage * 1000

    except ValueError:

        wage = 0

    return wage   

df['Wages'] = df['Wage'].apply(Wage_to_float)

print(df['Wages'])
print('Least 10 CLubs with Market Value')

print(df.groupby('Club')['Value_float'].sum().sort_values(ascending = True).head(10))

print(df.groupby('Club')['Value_float'].sum().sort_values(ascending = False).head(10))
plt.figure(figsize = (25,16))

sns.scatterplot(df['Value_float']/1000000 , df['Wages'], hue = df['Preferred Foot'])

plt.ylim(0,600000)
print('Top 10 Players with their Age and Club Names')

print(df.groupby(['Name','Age','Club'])['Overall'].max().sort_values(ascending = False).head(10))
print('Clubs with their total numbers of Players')

print(df.groupby('Club')['Name'].count().sort_values(ascending = False))
print('Top 10 team with best players')

print(df.groupby(['Club'])['Overall'].mean().sort_values(ascending = False).head(10))
print(df.groupby('Club')['Overall'].mean().sort_values(ascending = True).head(10))
x = df['Age']

Agex = sns.distplot(x , bins = 60 , kde = True ,color = 'g')

Agex.set_title('Numbers of players according to their age')

Agex.set_xlabel('Age')

Agex.set_ylabel('Numbers of Players')

plt.show()

value = df['Value_float']

sns.regplot(x = value/1000000, y = 'Overall', data = df , fit_reg = False)

plt.show()
agepot = sns.regplot(x = 'Age', y = 'Potential',fit_reg = True ,data = df )

agepot.set_title('Relation between Age and Potential of the Player',fontsize = 16)

plt.show()
AOax = sns.violinplot(x = 'Age', y = 'Overall', data = df)

AOax.set_title('Relation between the Age and Overall')

plt.xticks(Rotation = 70)

plt.show()
SS_mean = df['SprintSpeed'].mean()

df['SprintSpeed'] = df['SprintSpeed'].fillna(SS_mean)

S_mean = df['Stamina'].mean()

df['Stamina'] = df['Stamina'].fillna(S_mean)



cols = ['Age','Overall','Potential','SprintSpeed','Stamina']

sns.pairplot(df[cols],height = 2.5)

plt.show()
stats.probplot(df['Overall'],plot = plt)

plt.show()
print('Numbers of Players according to their Position')

print(df.groupby(['Position'])['Name'].nunique().head(28))
player_features = (

    'Acceleration', 'Aggression', 'Agility', 

    'Balance', 'BallControl', 'Composure', 

    'Crossing', 'Dribbling', 'FKAccuracy', 

    'Finishing', 'GKDiving', 'GKHandling', 

    'GKKicking', 'GKPositioning', 'GKReflexes', 

    'HeadingAccuracy', 'Interceptions', 'Jumping', 

    'LongPassing', 'LongShots', 'Marking', 'Penalties'

)

for i,val in df.groupby(df['Position'])[player_features].mean().iterrows():

    print('Position {}: {}, {}, {}'.format(i , *tuple(val.nlargest(3).index)))