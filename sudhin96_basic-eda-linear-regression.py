import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
train = pd.read_csv('../input/train_V2.csv')
train.shape
train.columns.values
train.isnull().sum()
train=train.dropna()
train.plot(x="kills",y="damageDealt", kind="scatter",figsize=(15,10))
head_shots=train.query('headshotKills>0 & headshotKills<20')

plt.figure(figsize=(15, 8))

plt.hist(head_shots['headshotKills'])

plt.title("Head Shots",fontsize=25)

plt.ylabel("Count",fontsize=15)

plt.xlabel("Shots",fontsize=15)



longest_kills=train[train["longestKill"]>100]

plt.figure(figsize=(15, 8))

plt.hist(longest_kills["longestKill"])

plt.title("Longest Kill",fontsize=25)

plt.ylabel("Count",fontsize=15)

plt.xlabel("Shots",fontsize=15)
plt.figure(figsize=(15, 8))

plt.hist(train["matchDuration"]/60,)

plt.title("Match Duration",fontsize=25)

plt.ylabel("Count",fontsize=15)

plt.xlabel("time per minuate",fontsize=15)
ride = train.query('rideDistance >0 & rideDistance <20000')

plt.figure(figsize=(15, 8))

plt.hist(ride['rideDistance'],bins=40)

plt.title("Ride Distance",fontsize=25)

plt.ylabel("Count",fontsize=15)

plt.xlabel("Distance",fontsize=15)
walk=train.query('walkDistance>0 & walkDistance<5000')

plt.figure(figsize=(15, 8))

plt.hist(walk['walkDistance'],bins=40)

plt.title("Walk Distance",fontsize=25)

plt.ylabel("Count",fontsize=15)

plt.xlabel("Distance",fontsize=15)
plt.figure(figsize=(15, 8))

f=sns.countplot(train["matchType"], saturation = 0.86,

              linewidth=2,

              edgecolor = sns.set_palette("dark", 3))

f.set_xticklabels(train["matchType"],rotation=45)

plt.title("Match Types",fontsize=25)

plt.ylabel("Count",fontsize=15)

plt.xlabel("Type",fontsize=15)





cols=train.columns

aggs = ['count','min','mean','max']

grp = train.loc[train['matchType'].str.contains('solo')].groupby('matchId')

grpSolo = grp[cols].sum()

grp = train.loc[train['matchType'].str.contains('duo')].groupby('matchId')

grpDuo = grp[cols].sum()

grp=train.loc[train['matchType'].str.contains('squad')].groupby('matchId')

grpTeam=grp[cols].sum()

pd.concat([grpSolo.describe().T[aggs],grpDuo.describe().T[aggs], grpTeam.describe().T[aggs]], keys=['solo', 'duo', 'team'], axis=1)
CategoryDamageDealt = pd.cut(train['damageDealt'], [-1, 0, 10, 50, 150, 300, 1000, 6000],

      labels = ['O Damage Taken', 

                '1-10 Damage Taken', 

                '11-50 Damage Taken', 

                '51-150 Damage Taken',

                '151-300 Damage Taken',

                '301-1000 Damage Taken',

                '1000+ Damage Taken'])

plt.figure(figsize=(16, 8))

sns.countplot(CategoryDamageDealt, saturation = 0.86,

              linewidth=2,

              edgecolor = sns.set_palette("dark", 3))

plt.title("Damage Death",fontsize=25)

plt.xlabel("Damage Taken",fontsize=15)

plt.ylabel("Number",fontsize=15)
plt.subplots(figsize=(20,15))

sns.heatmap(train.corr(), 



    vmin=-1, vmax=1, center=0,



    cmap=sns.diverging_palette(20, 220, n=200),



    annot=True,



)
train.columns
from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split
target_data=train.drop(train.columns[[0, 1, 2,15]], axis=1)
X = target_data.drop('winPlacePerc', 1)

y = target_data['winPlacePerc']



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
lm = LinearRegression()

lm.fit(X_train,y_train)
predictions = lm.predict(X_test)
plt.figure(figsize=(16, 8))

plt.title("predict_data vs test_data",fontsize=25)

plt.scatter(y_test,predictions)
model = LinearRegression(normalize=True, n_jobs=8)



lreg = model.fit(X_train, y_train)
train_score=lreg.score(X_train, y_train)

print(train_score)
test_score=lreg.score(X_test,y_test)

print(test_score)