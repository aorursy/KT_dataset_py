

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns  # visualization tool

import os



for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/international-football-results-from-1872-to-2017/results.csv')

print(data.columns)

print(data.info())
print(data.city.unique())

print(data.country.unique())

print(data.tournament.unique())


print(np.max(data['home_score']))

print(np.max(data['away_score']))
print(data.corr())
x=data[(data['home_team']=='Turkey')&((data['home_score'])+data['away_score']>=3)]

x.info()

turkeyTotalMatch=data[(data['home_team']=='Turkey')|(data['away_team']=='Turkey')]

print("\n\n türkiyenin toplam oynadığı maç sayısı \n\n")

turkeyTotalMatch.info('RangeIndex')
Tf,ax= plt.subplots(figsize=(10,10))

sns.heatmap(x.corr(),annot=True,lineWidths=.5,fmt='.2f',ax=ax)

plt.show()


data.home_score.plot(kind='line',color='g',label='homeScore',lineWidth=4,alpha=0.5,grid=True,linestyle=':',figsize=(20, 20))

data.away_score.plot(color='r',label='awayScore',linewidth=4,alpha=0.5,grid=True,linestyle=':')

figsize=(20, 50)

plt.legend(loc='upper right')

plt.xlabel("x axis")

plt.ylabel("y axis")

plt.title("Line plot")

plt.show()
data.plot(kind='scatter',x='home_score',y='away_score',alpha=1,color='red',figsize=(15,10))



plt.xlabel='home_score'

plt.ylabel='away_score'

plt.title='scatter plot'
data.home_score.plot(kind='hist',bins=60,figsize=(12,12),color='g')

data.away_score.plot(kind='hist',bins=60,color='r',alpha=0.5)

plt.show()