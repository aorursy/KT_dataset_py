# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
players=pd.read_csv('../input/Players.csv')

Seasons=pd.read_csv('../input/Seasons_Stats.csv')
players.head()
Seasons.head()
players.shape #shape of the dataframe
players.isnull().sum()    #checking data quality
players.drop('Unnamed: 0' ,axis=1,inplace=True)
players.dropna(how='all',inplace=True) #dropping the player whose value is null
players.set_index('Player',inplace=True) #setting the player name as the dataframe indexx

players.head(2)
print('The Tallest Player in NBA History is:',players['height'].idxmax(),' with height=',players['height'].max(),' cm')

print('The Heaviest Player in NBA History is:',players['weight'].idxmax(),' with weight=',players['weight'].max(),' kg')
print('The Shortest Player in NBA History is:',players['height'].idxmin(),' with height=',players['height'].min(),' cm')

print('The Lightest Player in NBA History is:',players['weight'].idxmin(),' with weight=',players['weight'].min(),' kg')
print('The average height of NBA Players is ',players['height'].mean())

print('The average weight of NBA Players is ',players['weight'].mean())
bins=range(150,250,10)

plt.hist(players["height"],bins,histtype="bar",rwidth=1.2,color='#0ff0ff')

plt.xlabel('Height in Cm')

plt.ylabel('Count')

plt.axvline(players["height"].mean(), color='b', linestyle='dashed', linewidth=2)

plt.plot()
bins=range(60,180,10)

plt.hist(players["weight"],bins,histtype="bar",rwidth=1.2,color='#4400ff')

plt.xlabel('Weight in Kg')

plt.ylabel('Count')

plt.axvline(players["weight"].mean(), color='black', linestyle='dashed', linewidth=2)

plt.plot()
college=players.groupby(['collage'])['height'].count().reset_index().sort_values(by='height',ascending=False)[:10]

college.set_index('collage',inplace=True)

college.columns=['Count']

ax=college.plot.bar(width=0.8)

for p in ax.patches:

    ax.annotate(format(p.get_height()), (p.get_x()+0.15, p.get_height()+0.35))

plt.show()
city=players.groupby(['birth_state'])['height'].count().reset_index().sort_values(by='height',ascending=False)[:10]

city.set_index('birth_state',inplace=True)

city.columns=['Count']

ax=city.plot.bar(width=0.8,color='#ab1abf')

for p in ax.patches:

    ax.annotate(format(p.get_height()), (p.get_x()+0.15, p.get_height()+1))

plt.show()
Seasons.drop('Unnamed: 0',axis=1,inplace=True)
Seasons.head()
Seasons.isnull().sum()
Seasons=Seasons[Seasons['Player'] !=0] #removing entries without names

players.reset_index(inplace=True)