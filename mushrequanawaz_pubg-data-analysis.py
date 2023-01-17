# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
data=pd.read_csv('/kaggle/input/pubg-finish-placement-prediction/train_V2.csv')

data.info()

data.describe()
data.shape

#changing the datatype
data=data.astype({'matchType':'category'})
data.info()
def pc(x,bins):
    counts, edges=np.histogram(x, bins=bins)
    pdf=counts/sum(counts)
    cdf=np.cumsum(pdf)
    plt.plot(edges[1:],pdf, label='Probabilty Density Function')
    plt.plot(edges[1:],cdf, label='Cumilative Density Function')
    plt.title('Probability and Cumulative density graphs')
    plt.legend(loc="right")
def discreet(x):
    print(x.value_counts())
    sns.boxplot(x,color='red')
    plt.show()
    pc(x,int(x.max()))
    plt.show()
    x.value_counts().plot(kind='pie', autopct='%0.2f')
    plt.show()
    
def continuous(x):
    sns.violinplot(x,color='green')
    plt.title('Violin plot of all the players')
    plt.show()
    sns.distplot(x,rug=True,color='orange')
    plt.title('Distplot for all the players')
    plt.show()
    pc(x,int(x.max()))
    plt.title('Distribution functions for all players')
    plt.show()

discreet(data['assists'])
discreet(data['DBNOs'])
discreet(data['headshotKills'])
discreet(data['heals'])
discreet(data['kills'])
discreet(data['killStreaks'])
discreet(data['maxPlace'])
discreet(data['numGroups'])
discreet(data['revives'])
discreet(data['teamKills'])
discreet(data['vehicleDestroys'])
discreet(data['weaponsAcquired'])
discreet(data['roadKills'])

continuous(data['damageDealt'])
continuous(data['killPlace'])
continuous(data['killPoints'])
continuous(data['longestKill'])
continuous(data['rankPoints'])

continuous(data['rankPoints'])
continuous(data['matchDuration'])
continuous(data['rideDistance'])


continuous(data['walkDistance'])
continuous(data['winPoints'])
data[['numGroups','maxPlace']].corr()

def players(x):
    sns.distplot(x,rug=True, color='orange')
    plt.title('Distplot for players')
players(data['damageDealt'])
players(data['DBNOs'])
players(data['kills'])
players(data['heals'])
players(data['longestKill'])

sns.boxplot(data['winPlacePerc'])

##KILL EVALUATION
print('Percentage of players who did not kill a single player={0:.02f}%'.format(len(data[data['kills']==0])/len(data)*100))
print('Maximum number of kills in a match=',data['kills'].max())
