# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from pandas import crosstab

import seaborn as sns

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
games=pd.read_csv('../input/ign.csv')
games.head()
games.isnull().sum()  #checking if there are any null values in the dataset
games.fillna('Unknown',inplace=1)
games.drop(['Unnamed: 0','url'],axis=1,inplace=1) #removed the unneeded url column

games.head(2)
print (games.shape)

print('\n The Various Platforms are :', games['platform'].unique(), 'Total: ',games['platform'].nunique())
print('Highest Score: ',games['score'].max() , '  Lowest Score: ' , games['score'].min())

print('\n The genres are: ', games['genre'].unique(), ' Total: ',games['genre'].nunique())
masterpiece=games[games['score']==10][['platform','genre']]

for i in masterpiece.columns:

    masterpiece[i].value_counts().plot.pie(autopct='%1.1f%%')

    fig = plt.gcf()

    fig.set_size_inches(6,6)

    plt.show()
ax=sns.countplot(games['release_year'])

for p in ax.patches:

    ax.annotate(format(p.get_height()), (p.get_x()-0.15, p.get_height()+20),fontsize=8)

plt.xticks(rotation=45)

plt.show()

games['score'].plot.hist()

plt.axvline(games['score'].mean(), color='b', linestyle='dashed', linewidth=2)

plt.xlabel('score')

plt.show()
years=games.groupby(['release_year','platform']).count()

years=years['genre'].reset_index()

years.head()