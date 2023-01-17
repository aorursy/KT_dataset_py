# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt # Visualization

import seaborn as sns # Visualization

from IPython.display import display, HTML # IPython notebook display

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os

# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/data.csv")

data.info()
data.head()
plt.rcParams['figure.figsize']=(25,16)

hm=sns.heatmap(data[['Age', 'Overall', 'Potential', 'Special',

    'Body Type', 'Position',

    'Height', 'Weight', 'Crossing',

    'Finishing', 'HeadingAccuracy', 'ShortPassing', 'Volleys', 'Dribbling',

    'Curve', 'FKAccuracy', 'LongPassing', 'BallControl', 'Acceleration',

    'SprintSpeed', 'Agility', 'Reactions', 'Balance', 'ShotPower',

    'Jumping', 'Stamina', 'Strength', 'LongShots', 'Aggression',

    'Interceptions', 'Positioning', 'Vision', 'Penalties', 'Composure',

    'Marking', 'StandingTackle', 'SlidingTackle']].corr(), annot = True, linewidths=.5, cmap='Blues')

hm.set_title(label='Heatmap of dataset', fontsize=20)
plt.xlabel('Height', fontsize=20)

plt.ylabel('Dribbling', fontsize=20)

plt.title('Height vs Dribbling', fontsize = 25)

sns.barplot(x='Height', y='Dribbling', data=data.sort_values('Height', inplace=False), alpha=0.6)
plt.xlabel('Weight', fontsize=20)

plt.ylabel('Dribbling', fontsize=20)

plt.title('Weight vs Dribbling', fontsize = 25)

sns.barplot(x='Weight', y='Dribbling', data=data.sort_values('Weight'),alpha=0.6)
skills = ['Overall', 'Potential', 'Crossing',

   'Finishing', 'HeadingAccuracy', 'ShortPassing', 'Volleys', 'Dribbling',

   'Curve', 'FKAccuracy', 'LongPassing', 'BallControl', 'Acceleration',

   'SprintSpeed', 'Agility', 'Reactions', 'Balance', 'ShotPower',

   'Jumping', 'Stamina', 'Strength', 'LongShots', 'Aggression',

   'Interceptions', 'Positioning', 'Vision', 'Penalties', 'Composure',

   'Marking', 'StandingTackle', 'SlidingTackle']
messi = data.loc[data['Name'] == 'L. Messi']

messi = pd.DataFrame(messi, columns = skills)

ronaldo = data.loc[data['Name'] == 'Cristiano Ronaldo']

ronaldo = pd.DataFrame(ronaldo, columns = skills)



plt.figure(figsize=(14,8))

sns.pointplot(data=messi,color='blue',alpha=0.6)

sns.pointplot(data=ronaldo, color='red', alpha=0.6)

plt.text(5,55,'Messi',color='blue',fontsize = 25)

plt.text(5,50,'Ronaldo',color='red',fontsize = 25)

plt.xticks(rotation=90)

plt.xlabel('Skills', fontsize=20)

plt.ylabel('Skill value', fontsize=20)

plt.title('Messi vs Ronaldo', fontsize = 25)

plt.grid()
display(

    HTML(data.sort_values('Overall', ascending=False)[['Name', 'Overall']][:10].to_html(index=False)

))
top_clubs = data.groupby(['Club'])['Overall'].max().sort_values(ascending = False)

top_clubs.head(5)
top_club_names = ('FC Barcelona', 'Juventus', 'Paris Saint-Germain', 'Chelsea', 'Manchester City')

clubs = data.loc[data['Club'].isin(top_club_names) & data['Age']]

fig, ax = plt.subplots()

fig.set_size_inches(20, 10)

ax = sns.boxenplot(x="Club", y="Age", data=clubs)

ax.set_title(label='Age distribution in the top 5 clubs', fontsize=25)

plt.xlabel('Clubs', fontsize=20)

plt.ylabel('Age', fontsize=20)

plt.grid()
countries_names = ('France', 'Brazil', 'Germany', 'Belgium', 'Spain', 'Netherlands', 'Argentina', 'Portugal', 'Chile', 'Colombia')

countries = data.loc[data['Nationality'].isin(countries_names) & data['Age']]

fig, ax = plt.subplots()

fig.set_size_inches(20, 10)

ax = sns.boxenplot(x="Nationality", y="Age", data=countries)

ax.set_title(label='Age distribution in countries', fontsize=25)

plt.xlabel('Countries', fontsize=20)

plt.ylabel('Age', fontsize=20)

plt.grid()