# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from matplotlib import pyplot as plt

import seaborn as sns  # visualization tool



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
futbol = pd.read_csv("../input/data.csv")
futbol.info()
futbol.head()
futbol.tail()
futbol.describe()
# shape of the dataset : ( 18207 rows and 89 columns )

futbol.shape
# How many null objects in the datasets

futbol.isnull().sum()
# How many columns are there

len(futbol.columns)
# How many rows are there

len(futbol.index)
# Random rows

futbol.sample(5)
futbol.hist('Age', figsize=(7,7))
futbol[['Name', 'Age', 'Wage', 'Value']].max()
futbol[['Name', 'Age', 'Wage', 'Value']].min()
futbol[futbol["Club"] == "Galatasaray SK"][['Name' , 'Position' , 'Overall' , 'Age', 'Wage', 'Nationality']].head(10)
futbol.sort_values(by = 'Age' , ascending = False)[['Name', 'Age', 'Club', 'Nationality']].set_index('Name').head()
futbol.sort_values(by = 'Age' , ascending = True)[['Name', 'Age', 'Club', 'Nationality']].set_index('Name').sample(10)
futbol.sort_values(by = 'Value' , ascending = False)[['Name', 'Age', 'Club', 'Nationality', 'Overall', 'Value', 'Wage']].head()
futbol.sort_values(by = 'ShotPower' , ascending = False)[['Name', 'Age', 'Club', 'Nationality', 'ShotPower']].head()
futbol.hist(bins=75, figsize=(25,6))
#correlation map

f,ax = plt.subplots(figsize=(25, 10))

hm = sns.heatmap(futbol[['Age', 'Overall', 'Potential', 'Value', 'Wage',

                'Acceleration', 'Aggression', 'Agility', 'Balance', 'BallControl', 

                'Body Type','Composure', 'Crossing','Dribbling', 'FKAccuracy', 'Finishing', 

                'HeadingAccuracy', 'Interceptions','International Reputation',

                'Joined', 'Jumping', 'LongPassing', 'LongShots',

                'Marking', 'Penalties', 'Position', 'Positioning',

                'ShortPassing', 'ShotPower', 'Skill Moves', 'SlidingTackle',

                'SprintSpeed', 'Stamina', 'StandingTackle', 'Strength', 'Vision',

                'Volleys']].corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)

hm.set_title(label='Heatmap of dataset', fontsize=20)

plt.show()
# Histogram: number of players's age

sns.set(style ="dark", palette="colorblind", color_codes=True)

x = futbol.Age

plt.figure(figsize=(12,8))

ax = sns.distplot(x, bins = 58, kde = False, color='g')

ax.set_xlabel(xlabel="Player\'s age", fontsize=16)

ax.set_ylabel(ylabel='Number of players', fontsize=16)

ax.set_title(label='Histogram of players age', fontsize=20)

plt.show()