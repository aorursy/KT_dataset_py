# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.metrics import mean_squared_error

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import train_test_split

from collections import Counter

from IPython.core.display import display, HTML

sns.set_style('darkgrid')
df = pd.read_csv('/kaggle/input/glass/glass.csv')

df.head()
corr = df.corr()

#Plot figsize

fig, ax = plt.subplots(figsize=(10, 8))

#Generate Heat Map, allow annotations and place floats in map

sns.heatmap(corr, cmap='coolwarm', annot=True, fmt=".2f")

#Apply xticks

plt.xticks(range(len(corr.columns)), corr.columns);

#Apply yticks

plt.yticks(range(len(corr.columns)), corr.columns)

#show plot

plt.show()
print (corr)
features = ['RI','Na','Mg','Al','Si','K','Ca','Ba','Fe','Type']

data = df[features]



# select target

target = data['Type']

data = data.drop('Type', axis=1)



X_train, X_test, y_train, y_test = train_test_split(

    data, target, test_size=0.4, random_state=0)

from sklearn.ensemble import GradientBoostingClassifier

#clf = GradientBoostingClassifier(loss='deviance', n_estimators=100, learning_rate=1.0,max_depth=2, random_state=0)

# Fit classifier with out-of-bag estimates

params = {'n_estimators': 1500, 'max_depth': 5, 'subsample': 0.5,

          'learning_rate': 0.01, 'min_samples_leaf': 1, 'random_state': 3}

clf = GradientBoostingClassifier(**params)

clf.fit(X_train, y_train)

predictions = clf.predict(X_test)