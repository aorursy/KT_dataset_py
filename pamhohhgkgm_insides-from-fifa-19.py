

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

#data manupulated



import matplotlib.pyplot as plt

import seaborn as sns

#data visualization



import os

print(os.listdir("../input"))
pd.set_option('display.max_columns',100)

pd.set_option('display.max_rows',100)

#defining display view of data
import datetime #for date

import geopandas as gpd #maps

import pycountry 
from math import pi

from IPython.display import display, HTML # for displaying in jypyter
data=pd.read_csv('../input/data.csv')
data.head()
data.info()
data.describe()
data.shape
data.nunique()
data.isnull().any()
data.columns

chosen_columns = [

    'Name',

    'Age',

    'Nationality',

    'Overall',

    'Potential',

    'Special',

    'Acceleration',

    'Aggression',

    'Agility',

    'Balance',

    'BallControl',

    'Body Type',

    'Composure',

    'Crossing',

    'Curve',

    'Club',

    'Dribbling',

    'FKAccuracy',

    'Finishing',

    'GKDiving',

    'GKHandling',

    'GKKicking',

    'GKPositioning',

    'GKReflexes',

    'HeadingAccuracy',

    'Interceptions',

    'International Reputation',

    'Jersey Number',

    'Jumping',

    'Joined',

    'LongPassing',

    'LongShots',

    'Marking',

    'Penalties',

    'Position',

    'Positioning',

    'Preferred Foot',

    'Reactions',

    'ShortPassing',

    'ShotPower',

    'Skill Moves',

     'SlidingTackle',

    'SprintSpeed',

    'Stamina',

    'StandingTackle',

    'Strength',

    'Value',

    'Vision',

    'Volleys',

    'Wage',

    'Weak Foot',

    'Work Rate'

]
df = pd.DataFrame(data, columns = chosen_columns)
df
plt.rcParams['figure.figsize']=(25,16)

hm=sns.heatmap(df[['Age', 'Overall', 'Potential', 'Value', 'Wage',

                'Acceleration', 'Aggression', 'Agility', 'Balance', 'BallControl', 

                'Body Type','Composure', 'Crossing','Dribbling', 'FKAccuracy', 'Finishing', 

                'HeadingAccuracy', 'Interceptions','International Reputation',

                'Joined', 'Jumping', 'LongPassing', 'LongShots',

                'Marking', 'Penalties', 'Position', 'Positioning',

                'ShortPassing', 'ShotPower', 'Skill Moves', 'SlidingTackle',

                'SprintSpeed', 'Stamina', 'StandingTackle', 'Strength', 'Vision',

                'Volleys']].corr(), annot = True, linewidths=.5, cmap='Greens')

hm.set_title(label='Heatmap of dataset', fontsize=20)

hm;
def make_scatter(df):

    feats = ('Agility', 'Balance', 'Dribbling', 'SprintSpeed')

    

    for index, feat in enumerate(feats):

        plt.subplot(len(feats)/4+1, 4, index+1)

        ax = sns.regplot(x = 'Acceleration', y = feat, data = df)



plt.figure(figsize = (20, 20))

plt.subplots_adjust(hspace = 0.4)



make_scatter(df)
sns.set(style ="dark", palette="colorblind", color_codes=True)

x = df.Age

plt.figure(figsize=(10,10))

ax = sns.distplot(x, bins = 58, kde = False, color='Orange')

ax.set_xlabel(xlabel="Players age", fontsize=16)

ax.set_ylabel(ylabel='Number of players', fontsize=16)

ax.set_title(label='Histogram of players age as per their age', fontsize=20)

plt.show()
some_clubs = ('Juventus', 'Real Madrid', 'Paris Saint-Germain', 'FC Barcelona', 'Legia Warszawa', 'Manchester United')

df_club = df.loc[df['Club'].isin(some_clubs) & df['Age']]



fig, ax = plt.subplots()

fig.set_size_inches(15, 10)

ax = sns.violinplot(x="Club", y="Age", data=df_club);

ax.set_title(label='Distribution of Players age in some clubs', fontsize=20);
data.head()
df.groupby(['Club'])['Age'].sum().sort_values(ascending = True).head(5)  #youngest team
df.groupby(['Club'])['Age'].sum().sort_values(ascending = False).head(5) #oldest team
# The best player per position

display(HTML(df.iloc[df.groupby(df['Position'])['Overall'].idxmax()][['Name','Position','Age']].to_html(index=False)))
player_features = (

    'Acceleration', 'Aggression', 'Agility', 

    'Balance', 'BallControl', 'Composure', 

    'Crossing', 'Dribbling', 'FKAccuracy', 

    'Finishing', 'GKDiving', 'GKHandling', 

    'GKKicking', 'GKPositioning', 'GKReflexes', 

    'HeadingAccuracy', 'Interceptions', 'Jumping', 

    'LongPassing', 'LongShots', 'Marking', 'Penalties'

)



# Top three features per position

for i, val in df.groupby(df['Position'])[player_features].mean().iterrows():

    print('Position {}: {}, {}, {}, {}'.format(i, *tuple(val.nlargest(4).index)))
# Which is Better from left-footed or rigth-footed players?

sns.lmplot(x = 'BallControl', y = 'Dribbling', data = df,

          scatter_kws = {'alpha':0.002},

          col = 'Preferred Foot');
def value_to_int(df_value):

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



df['Value_float'] = df['Value'].apply(value_to_int)
df.groupby(['Club'])['Value_float'].sum().sort_values(ascending = False)  #club price as per release clause
df.groupby(['Club'])['Overall'].max().sort_values(ascending = False)