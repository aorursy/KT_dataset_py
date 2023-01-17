# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt # data visualization 



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df=pd.read_csv("../input/fifa19/data.csv")
df.columns
df.drop(['Unnamed: 0','Photo','Flag','Club Logo'],axis=1,inplace=True)
# checking for null values

df.isnull().sum()
# summary statistics

columns_of_interest = ['Age','Overall','Nationality','Club'] 

df[columns_of_interest].describe(include='all')
# countries with the most players in FIFA 19 

df['Nationality'].value_counts().head(20)
# Number of countries 

print('Total number of countries : {0}'.format(df['Nationality'].nunique()))
# Number of clubs

print('Total number of clubs : {0}'.format(df['Club'].nunique()))
# players with 90 or more Overall 

df[df['Overall'] >= 90][['Name','Age','Overall']]
# analyzing the top players from each attribute

attributes = ['Crossing', 'Finishing', 'HeadingAccuracy', 'ShortPassing', 'Volleys',

       'Dribbling', 'Curve', 'FKAccuracy', 'LongPassing', 'BallControl',

       'Acceleration', 'SprintSpeed', 'Agility', 'Reactions', 'Balance',

       'ShotPower', 'Jumping', 'Stamina', 'Strength', 'LongShots',

       'Aggression', 'Interceptions', 'Positioning', 'Vision', 'Penalties',

       'Composure', 'Marking', 'StandingTackle', 'SlidingTackle', 'GKDiving',

       'GKHandling', 'GKKicking', 'GKPositioning', 'GKReflexes']

i=0

while i < len(attributes):

    print('Best {0} : {1}'.format(attributes[i],df.loc[df[attributes[i]].idxmax()][1]))

    i += 1
# creating a free agent category under Club column for players without a club

df['Club'].fillna('Free Agent',inplace = True)

# best 20 free agents 

df[df['Club'] == 'Free Agent'][['Name','Overall']].head(20)
# age distribution of players

df['Age'].plot(kind = 'hist',bins= 200, color= 'red',label = 'Age',linewidth=500,alpha = 1.0,grid = True,linestyle = '-')

plt.legend

plt.xlabel('Age')

plt.ylabel('# of Players')

plt.title('Average Age')

plt.show()
# average age 

df['Age'].mean()
# player counts by position

df['Position'].value_counts() 
# position frequency pie chart

values = df['Position'].value_counts().values

plt.pie(values,labels=(df['Position'].value_counts().index),autopct='%1.1f%%', pctdistance=1.3)

plt.axis('equal')

plt.show()
# best youngsters by position

young_team_df = df.where(df['Age']<=22)

young_team_list = young_team_df.groupby('Position')['Overall'].idxmax()

young_team = young_team_df.iloc[young_team_list]

young_team[['Position','Name','Overall']]