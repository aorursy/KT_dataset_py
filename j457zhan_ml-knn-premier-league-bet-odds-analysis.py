# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))

import matplotlib.pyplot as plt

# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/PL.csv')

df.head()
df = df[['Season','Home', 'Away', 

         'Home_First', 'Away_First', 'Result1_First','Result2_First',

         'Home_Second', 'Away_Second', 'Result1_Second','Result2_Second',

         'Home_Final', 'Away_Final', 'Result1_Final','Result2_Final', 

         'Home_Pin', 'Draw_Pin', 'Away_Pin', 

         'Home_Bet','Draw_Bet', 'Away_Bet', 

         'O_Pin', 'U_Pin', 'O_Bet', 'U_Bet']]

df = df.dropna()
from sklearn.model_selection import train_test_split

X = df[['Home_Pin', 'Draw_Pin', 'Away_Pin', 'O_Pin', 'U_Pin']]

y1 = df['Result1_Final']

y2 = df['Result2_Final']

X1_train, X1_test, y1_train, y1_test = train_test_split(X, y1, random_state=0)

X2_train, X2_test, y2_train, y2_test = train_test_split(X, y2, random_state=0)
from sklearn.neighbors import KNeighborsClassifier

k_range = range(1, 100)

scores = []



for k in k_range:

    knn = KNeighborsClassifier(n_neighbors = k)

    knn.fit(X1_train, y1_train)

    scores.append(knn.score(X1_test, y1_test))



plt.figure()

plt.xlabel('k')

plt.ylabel('accuracy')

plt.scatter(k_range, scores)
print('The most accurate prediction happens at {} neighbors with accuracy {}'.format(scores.index(max(scores)), max(scores)))
k_range = range(1, 100)

scores = []



for k in k_range:

    knn = KNeighborsClassifier(n_neighbors = k)

    knn.fit(X2_train, y2_train)

    scores.append(knn.score(X2_test, y2_test))



plt.figure()

plt.xlabel('k')

plt.ylabel('accuracy')

plt.scatter(k_range, scores)
print('The most accurate prediction happens at {} neighbors with accuracy {}'.format(scores.index(max(scores)), max(scores)))
t = [0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2]

knn = KNeighborsClassifier(n_neighbors = 92)

plt.figure()

for s in t:

    scores = []

    for i in range(1,10):

        X_train, X_test, y1_train, y1_test = train_test_split(X, y1, test_size=1-s)

        knn.fit(X_train, y1_train)

        scores.append(knn.score(X_test, y1_test))

        

    plt.plot(s, np.mean(scores), 'bo')
PL2018 = df.loc[df['Season'] == '2018/2019']

PL2017 = df.loc[df['Season'] == '2017/2018']

PL2016 = df.loc[df['Season'] == '2016/2017']

PL2015 = df.loc[df['Season'] == '2015/2016']

PL2014 = df.loc[df['Season'] == '2014/2015']

PL2013 = df.loc[df['Season'] == '2013/2014']

PL2012 = df.loc[df['Season'] == '2012/2013']

PL2011 = df.loc[df['Season'] == '2011/2012']

PL2010 = df.loc[df['Season'] == '2010/2011']

PL2009 = df.loc[df['Season'] == '2009/2010']

PL2008 = df.loc[df['Season'] == '2008/2009']

PL2007 = df.loc[df['Season'] == '2007/2008']

PL2018.head()
# 12-Neighbor for 2018/2019 0.5640256

from sklearn import preprocessing

from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier

from sklearn import metrics



# Premier League 2019

X_bet2018 = PL2018[['Home_Bet', 'Draw_Bet', 'Away_Bet', 'O_Bet', 'U_Bet']].values

y1_bet2018 = PL2018['Result1_Final'].values

y2_bet2018 = PL2018['Result2_Final'].values



X_bet2018 = preprocessing.StandardScaler().fit(X_bet2018).transform(X_bet2018.astype(float))

X_bet2018_train, X_bet2018_test, y1_bet2018_train, y1_bet2018_test = train_test_split(X_bet2018, y1_bet2018, test_size = 0.2, random_state=1)



Ks = 50

mean_acc = np.zeros((Ks -1))

std_acc = np.zeros((Ks - 1))

ConfusionMx = [];

for n in range(1, Ks):

    neigh = KNeighborsClassifier(n_neighbors = n).fit(X_bet2018_train, y1_bet2018_train)

    yhat = neigh.predict(X_bet2018_test)

    

    mean_acc[n-1] = metrics.accuracy_score(y1_bet2018_test, yhat)

    

    

mean_acc

plt.plot(range(1,Ks), mean_acc)
print('The most accurate prediction happens at {} neighbors with accuracy {}'.format(np.where(mean_acc==mean_acc.max())[0][0], mean_acc.max()))
X_bet2017 = PL2017[['Home_Bet', 'Draw_Bet', 'Away_Bet', 'O_Bet', 'U_Bet']].values

y1_bet2017 = PL2017['Result1_Final'].values

y2_bet2017 = PL2017['Result2_Final'].values



X_bet2017 = preprocessing.StandardScaler().fit(X_bet2017).transform(X_bet2017.astype(float))

X_bet2017_train, X_bet2017_test, y1_bet2017_train, y1_bet2017_test = train_test_split(X_bet2017, y1_bet2017, test_size = 0.2, random_state=1)



Ks = 50

mean_acc = np.zeros((Ks -1))

std_acc = np.zeros((Ks - 1))

ConfusionMx = [];

for n in range(1, Ks):

    neigh = KNeighborsClassifier(n_neighbors = n).fit(X_bet2017_train, y1_bet2017_train)

    yhat = neigh.predict(X_bet2017_test)

    

    mean_acc[n-1] = metrics.accuracy_score(y1_bet2017_test, yhat)

    

    

mean_acc

plt.plot(range(1,Ks), mean_acc)
print('The most accurate prediction happens at {} neighbors with accuracy {}'.format(np.where(mean_acc==mean_acc.max())[0][0], mean_acc.max()))