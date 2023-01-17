# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt
import seaborn as sns

pubg_set = pd.read_csv("/kaggle/input/pubg-finish-placement-prediction/train_V2.csv")
pubg_set
pubg_set.shape
pubg_set.info()
pubg_set.isnull().sum()
pubg_set[pubg_set["Id"].duplicated()]
pubg_set.columns
#win_percentage 
sns.boxplot(pubg_set['winPlacePerc'])
pubg_set=pubg_set.astype({'matchType':'category'})
pubg_set['matchType'].value_counts().plot(kind='pie', autopct='%0.1f')
pubg_set=pubg_set.astype({'assists':'category'})
pubg_set['assists'].value_counts().plot(kind='pie', autopct='%0.2f')
counts, edges=np.histogram(pubg_set['assists'], bins=15)
pdf=counts/sum(counts)
cdf=np.cumsum(pdf)
plt.plot(edges[1:],pdf, label='Probabilty Density Function')
plt.plot(edges[1:],cdf, label='Cumilative Density Function')
plt.title('Probability and Cumulative density graphs')
plt.legend(loc="right")
##Univariante Analysis
def pdf_CDF(xi,bins):
    counts, edges=np.histogram(xi, bins=bins)
    pdf=counts/sum(counts)
    cdf=np.cumsum(pdf)
    plt.plot(edges[1:],pdf, label='Probabilty Density Function')
    plt.plot(edges[1:],cdf, label='Cumilative Density Function')
    plt.title('Probability and Cumulative density graphs')
    plt.legend(loc="right")
def ad(x):
        x.value_counts().plot(kind='pie', autopct='%0.2f')
        plt.show()
        pdfcdf(x,int(x.max()))
        plt.show()
def ac(x):
    sns.distplot(x,rug=True)
    plt.title('Distplot for all the players')
    plt.show()
    pdfcdf(x,int(x.max()))
    plt.title('Distribution functions for all players')
    plt.show()
#acessing assists
plt.hist(pubg_set['assists'], bins=22)

#acessing boots
ad(pubg_set['boosts'])
plt.hist(pubg_set['boosts'], bins=33)
#acessing damagedealt
ac(pubg_set['damageDealt'])
#acessing knockouts
ad(pubg_set['DBNOs'])
#accessing headshot

ad(pubg_set['headshotKills'])
#acessing heals
ad(pubg_set['heals'])
#acessing killplace
sns.boxplot(pubg_set['killPlace'])
plt.show()
pdfcdf(pubg_set['killPlace'],100)
#acessing killPoints
ac(pubg_set['killPoints'])
#acessing killstreak
ad(pubg_set['killStreaks'])
#Acessing longestkill
ac(pubg_set['longestKill'])
#acessing matchduration
ac(pubg_set['matchDuration'])
#acessing the kills
ad(pubg_set['kills'])
#acessing Maxplace
ad(pubg_set['maxPlace'])
#acessing numgroups
ad(pubg_set['numGroups'])
#acessing rankpoints
ac(pubg_set['rankPoints'])
#acessing revives
ad(pubg_set['revives'])
#acessing ridedistance
ac(pubg_set['rideDistance'])
#Conclusion: from the above analysis I can conclude that most players get eliminated or gets out in the early duration of the match.
#Multivariant Analysis

f,ax = plt.subplots(figsize=(10, 10))
sns.heatmap(pubg_set.corr(), annot=True, linewidths=.5, fmt= '.1f', label='Correlation between variables',ax=ax)
f,ax = plt.subplots(figsize=(10, 10))
print('HeatMap for the players who have won the match:')
sns.heatmap(pubg_set[pubg_set['winPlacePerc']==1].corr(), annot=True, linewidths=.5, fmt= '.1f', label='Correlation between variables',ax=ax)