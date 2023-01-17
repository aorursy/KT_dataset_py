# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



import matplotlib.pyplot as plt; plt.rcdefaults()

import numpy as np

import matplotlib.pyplot as plt

 



# Any results you write to the current directory are saved as output.
#Reading the data

fifa_df=pd.read_csv('../input/data.csv')

fifa_df.head()
#checking the type of each column

fifa_df.info()
#checking the no.of rows and columns

fifa_df.shape
fifa_df.describe()
# Spread of players age

sns.set(style ="dark",color_codes=True)

x = fifa_df.Age

plt.figure(figsize=(12,8))

ax = sns.distplot(x, bins = 30, kde = False, color='b')

ax.set_xlabel(xlabel="Age", fontsize=16)

ax.set_ylabel(ylabel='Count of players', fontsize=16)

ax.set_title(label='Distribution of players age', fontsize=20)

plt.show()

# Checking the 10 youngest players club and nationality wise

eldest = fifa_df.sort_values('Age', ascending = True)[['Name', 'Nationality','Club', 'Age']].head(10)

eldest.set_index('Name', inplace=True)

print(eldest)

  
# Checking the 10 oldest players club and nationality wise

eldest = fifa_df.sort_values('Age', ascending = False)[['Name', 'Nationality','Club', 'Age']].head(10)

eldest.set_index('Name', inplace=True)

print(eldest)

  
# Age wise oldest team

fifa_df.groupby(['Club'])['Age'].sum().sort_values(ascending = False).head(5)

# Age wise oldest nationality

fifa_df.groupby(['Nationality'])['Age'].sum().sort_values(ascending = False).head(5)

# Club wise check the top 10 total potential

fifa_df.groupby(['Club'])['Potential'].sum().sort_values(ascending = False).head(10)

# Top 5 left-footed players who are not GK



df1 = fifa_df[(fifa_df['Preferred Foot'] == 'Left') & (fifa_df.Position != 'GK')]

df1[['Name','Overall','Club','Nationality']].head()
# Top 5 right-footed players who are not GK



df1 = fifa_df[(fifa_df['Preferred Foot'] == 'Right') & (fifa_df.Position != 'GK')]

df1[['Name','Overall','Club','Nationality']].head()
# Top 5 Right footed GK



df1 = fifa_df[(fifa_df['Preferred Foot'] == 'Right') & (fifa_df.Position == 'GK')]

df1[['Name','Overall','Club','Nationality']].head()
# Top 5 Left footed GK



df1 = fifa_df[(fifa_df['Preferred Foot'] == 'Left') & (fifa_df.Position == 'GK')]

df1[['Name','Overall','Club','Nationality']].head()
# Best player in each postition

fifa_df.iloc[fifa_df.groupby(fifa_df['Position'])['Overall'].idxmax()][['Name', 'Position','Club','Nationality','Overall']]
# Lets check age wise agility of players

fifa_df.plot(kind="scatter" , x="Age", y ="Agility", color="blue", alpha=0.5)

plt.xlabel("Age")      

plt.ylabel("Agility")

plt.title("Age vs Agility")

plt.show()
#checking the reputation of players internationally

sns.countplot(x='International Reputation', data=fifa_df)

# check special ability vs skill moves

x=fifa_df['Skill Moves']

y=fifa_df['Special']

plt.hexbin(x, y, gridsize=(30,30))

plt.show()

 