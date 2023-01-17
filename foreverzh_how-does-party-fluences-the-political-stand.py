% matplotlib inline

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import warnings

warnings.filterwarnings("ignore")



cts = pd.read_csv('../input/cts.csv')

votes = pd.read_csv('../input/votes.csv')
print(cts.info())

print(votes.info())

votes =  votes.dropna()
sns.set(style="ticks", color_codes=True)

sns.distplot(cts['trump_score'], kde = False)
len(cts[cts['trump_score'] == 1])+ len(cts[cts['trump_score'] == 0])
len(cts[cts.party == ' I'])
cts = cts[cts.party != ' I']
fig, axes = plt.subplots(2, 1, sharex = True)

sns.distplot(cts[cts['party'] == ' R']['trump_score'], color = 'r', kde = False, ax = axes[0])

sns.distplot(cts[cts['party'] == ' D']['trump_score'], color = 'r', kde = False, ax = axes[1])

axes[0].set_xlabel('Distribution of Republic')

axes[1].set_xlabel('Distribution of Democracy')
g = sns.FacetGrid(cts, col = 'party', row = 'chamber')

g.map(sns.distplot, 'trump_score')
cts.loc[cts.party == ' R', 'party'] = 1

cts.loc[cts.party == ' D', 'party'] = 2
features = []

for i in cts.trump_margin:

    feature = []

    feature.append(i)

    features.append(feature)
from sklearn import linear_model

from sklearn.cross_validation import train_test_split

feature_train, feature_test, res_train, res_test = train_test_split(

         features, cts.trump_score, test_size=0.3, random_state=42)

reg = linear_model.LinearRegression()

reg.fit(feature_train, res_train)
fig = plt.figure()

plt.scatter(feature_test,res_test)

plt.plot(feature_test, reg.predict(feature_test), color = 'r')
reg.score(feature_test, res_test)
features = []

for i in cts.party:

    feature = []

    feature.append(i)

    features.append(feature)

    

from sklearn import linear_model

from sklearn.cross_validation import train_test_split

feature_train, feature_test, res_train, res_test = train_test_split(

         features, cts.trump_score, test_size=0.3, random_state=42)

reg = linear_model.LinearRegression()

reg.fit(feature_train, res_train)



fig = plt.figure()

plt.scatter(feature_test,res_test)

plt.plot(feature_test, reg.predict(feature_test), color = 'r')



reg.score(feature_test, res_test)
features = []

for i in zip(cts.trump_margin, cts.party):

    feature = []

    for j in range(len(i)):

        feature.append(i[j])

    features.append(feature)

from sklearn.cross_validation import train_test_split

feature_train, feature_test, res_train, res_test = train_test_split(

         features, cts.trump_score, test_size=0.3, random_state=42)
from sklearn import linear_model

reg = linear_model.LinearRegression()

reg.fit(feature_train, res_train)
reg.score(feature_test, res_test)