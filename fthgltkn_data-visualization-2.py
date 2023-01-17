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
NBA_player_of_week=pd.read_csv('../input/NBA_player_of_the_week.csv')

#NBA_player_of_week=pd.DataFrame(NBA_player_of_week)
NBA_player_of_week.head()
player_frequency=NBA_player_of_week.Player.value_counts()

player_frequency

#NBA_player_of_week.Player.unique()
data=pd.DataFrame({'NBA_player_of_week.Player':NBA_player_of_week.Player.unique(),'player_frequency.values':player_frequency.values})

new_index=(data['player_frequency.values'].value_counts(ascending=False)).index.values

sorted_data1=data.reindex(new_index)

sorted_data1.head(10)
sns.barplot(y=sorted_data1['NBA_player_of_week.Player'],x=sorted_data1['player_frequency.values'])

#plt.figure(figsize=(25,25))

plt.xlabel('player')

plt.ylabel('values')

plt.xticks(rotation=0)

plt.title('week of player values')

plt.show()

NBA_player_of_week.corr()
f,ax=plt.subplots(figsize=(15,10))

sns.heatmap(NBA_player_of_week.corr(),annot=True)

plt.show()
NBA_player_of_week.columns
NBA_player_of_week['Seasons in league'].plot(kind='line',color='blue',label='Seasons in league',grid='True',linestyle=':',alpha=0.5)

NBA_player_of_week['Age'].plot(kind='line',color='green',label='Age',grid='True',linestyle='-.',alpha=0.5)

plt.legend(loc='upper right')

plt.xlabel('x axis')

plt.ylabel('y axis')

plt.title('Line Plot')

plt.show()