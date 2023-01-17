import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import matplotlib

from scipy.stats import skew

from scipy.stats import boxcox

import seaborn as sns

sns.set_style('darkgrid')

matplotlib.rc('font', size=10)

matplotlib.rc('axes', titlesize=10)

matplotlib.rc('axes', labelsize=10)

matplotlib.rc('xtick', labelsize=10)

matplotlib.rc('ytick', labelsize=10)

matplotlib.rc('legend', fontsize=10)

matplotlib.rc('figure', titlesize=10)

%matplotlib inline

from sklearn.ensemble import RandomForestClassifier

from sklearn import preprocessing

from sklearn.model_selection import train_test_split

from sklearn.cross_validation import StratifiedKFold

from sklearn.model_selection import GridSearchCV

import math

import xgboost as xgb
data = pd.read_csv('../input/glass.csv')

data.head()
matplotlib.rc('font', size=20)

matplotlib.rc('axes', titlesize=20)

matplotlib.rc('axes', labelsize=20)

matplotlib.rc('xtick', labelsize=20)

matplotlib.rc('ytick', labelsize=20)

matplotlib.rc('legend', fontsize=20)

matplotlib.rc('figure', titlesize=20)

train = data.drop('Type', axis=1)

corr = train.corr()

cmap = sns.diverging_palette(220, 220, as_cmap=True)

plt.figure(figsize=(20,10))

sns.heatmap(corr, cmap=cmap)
X = data.drop(['Type'], axis=1)

Y = data['Type']

data.corr()['Type'].abs().sort_values(ascending= False)
classes = X.columns.values

X_u = pd.DataFrame()

for c in classes:

    scaled = preprocessing.scale(X[c]) 

    boxcox_scaled = preprocessing.scale(boxcox(X[c] + np.max(np.abs(X[c]) +1))[0])

    X_u[c] = boxcox_scaled

    skness = skew(scaled)

    boxcox_skness = skew(boxcox_scaled) 

    figure = plt.figure()

    figure.add_subplot(121)   

    plt.hist(scaled,facecolor='blue',alpha=0.5) 

    plt.xlabel(c + " - Transformed") 

    plt.title("Skewness: {0:.2f}".format(skness)) 

    figure.add_subplot(122) 

    plt.hist(boxcox_scaled,facecolor='red',alpha=0.5) 

    plt.title("Skewness: {0:.2f}".format(boxcox_skness)) 

    plt.show()
X = X_u

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, random_state=42)

rf = RandomForestClassifier(max_features='auto',n_jobs=-1, random_state=1)

params = { "criterion" : ["gini", "entropy"]

              , "min_samples_leaf" : [1, 5, 10]

              , "min_samples_split" : [2, 4, 10, 12, 16]

              , "n_estimators": [100, 125, 200]

         }

GS = GridSearchCV(estimator=rf, param_grid=params, cv=5,n_jobs=-1)

GS= GS.fit(X_train,Y_train)

print(GS.best_score_)

print(GS.best_params_)
rf = RandomForestClassifier(criterion='gini', n_estimators=100, min_samples_leaf=1, min_samples_split=4, random_state=1,n_jobs=-1)

rf.fit(X_train,Y_train)

pred = rf.predict(X_test)

rf.score(X_test,Y_test)