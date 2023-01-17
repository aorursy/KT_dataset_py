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


import random as rnd



# visualization

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



# machine learning

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import Perceptron

from sklearn.linear_model import SGDClassifier

from sklearn.tree import DecisionTreeClassifier
filename='/kaggle/input/top50spotify2019/top50.csv'

T=pd.read_csv(filename,encoding='ISO-8859-1')

T.head()

filename='/kaggle/input/top50spotify2019/top50.csv'

spoti=pd.read_csv(filename,encoding='ISO-8859-1')

spoti.head(10)
T.isnull().sum()

T.fillna(1)

#Calculating the number of songs by each of the artists

print(T.groupby('Artist.Name').size())

popular_Artist=T.groupby('Artist.Name').size()

print(popular_Artist)

Artist_list=T['Artist.Name'].values.tolist()
print(type(T['Track.Name']))

popular_genre=T.groupby('Track.Name').size().unique

print(popular_genre) 

genre_list=T['Track.Name'].values.tolist()
print(type(T['Length.']))

popular_genre=T.groupby('Length.').size().unique

print(popular_genre) 

genre_list=T['Length.'].values.tolist()
print(type(T['Genre']))

popular_genre=T.groupby('Genre').size().unique

print(popular_genre)

genre_list=T['Genre'].values.tolist()
skew=T.skew()

print(skew)

# Removing the skew by using the boxcox transformations

transform=np.asarray(T[['Liveness']].values)

# Plotting a histogram to show the difference 

plt.hist(T['Liveness'],bins=10) #original data

plt.show()

plt.show()
xtick = ['dance pop', 'pop', 'latin', 'edm', 'canadian hip hop',

'panamanian pop', 'electropop', 'reggaeton flow', 'canadian pop',

'reggaeton', 'dfw rap', 'brostep', 'country rap', 'escape room',

'trap music', 'big room', 'boy band', 'pop house', 'australian pop',

'r&b en espanol', 'atl hip hop']

length = np.arange(len(xtick))

genre_groupby = T.groupby('Genre')['Track.Name'].agg(len)

plt.figure(figsize = (15,7))

plt.bar(length, genre_groupby)

plt.xticks(length,xtick)

plt.xticks(rotation = 90)

plt.xlabel('Genre', fontsize = 20)

plt.ylabel('Count of the tracks', fontsize = 20)

plt.title('Genre vs Count of the tracks', fontsize = 25)

f, ax = plt.subplots(figsize=(10,8))

x = spoti['Loudness..dB..']

ax = sns.distplot(x, bins=10)

plt.show()
pd.set_option('display.width', 100)

pd.set_option('precision', 3)

correlation=T.corr(method='spearman')

print(correlation)
# heatmap of the correlation 

plt.figure(figsize=(20,20))

plt.title('Correlation heatmap')

sns.heatmap(correlation,annot=True,vmin=-3,vmax=1,cmap="GnBu_r",center=3)
threshold = sum(spoti.Energy)/len(spoti.Energy)

print(threshold)

spoti["Energy_level"] = ["energized" if i > threshold else "without energy" for i in spoti.Energy]

spoti.loc[:10,["Energy_level","Energy"]]

#This caught my attention to the effect of energy level on music in here and i calcuted it. It classified according to mean of value
T.plot(kind='box', subplots=True)

plt.gcf().set_size_inches(15,15)

plt.show()