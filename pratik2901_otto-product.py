import pandas as pd

import numpy as np

from matplotlib import pyplot as plt

%matplotlib inline

import seaborn as sns

from sklearn.utils import shuffle
train_csv = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
train_csv = shuffle(train_csv)

train_csv=train_csv
train_csv.head()
order = sorted(set(train_csv['target']))

sns.countplot(x='target', data=train_csv,order=order)

plt.grid()

plt.title("No of Product of Each Class")

plt.figure(num=None, figsize=(20, 30), dpi=80, facecolor='w', edgecolor='k')
wt = train_csv.sum()

wt.drop(['target','id']).sort_values().plot(kind='barh', figsize=(15,20))

plt.grid()

plt.title("Weight Of Features")
df = train_csv.drop(['id','target'],axis=1).corr()

sns.heatmap(df)

plt.title("Correation Analysis")
df.var().sort_values().plot(kind='barh', figsize=(15,20))

plt.grid()

plt.title("Covariance Analysis")
train_csv.describe()
X = train_csv

Y = train_csv['target']

del X['target']

del X['id']
X.describe()
from sklearn import preprocessing

le = preprocessing.LabelEncoder()

le.fit(Y.values.tolist())

label=le.transform(Y)

print(list(le.classes_))

print(label)
noOfFeature = 45
from sklearn.feature_selection import RFE

from sklearn.ensemble import RandomForestClassifier

import timeit

start = timeit.default_timer()

clf = RandomForestClassifier()

rfe = RFE(clf, noOfFeature)

fit = rfe.fit(X, label)

print("Time take %.2f "%(timeit.default_timer()-start))

print(("Num Features: %d") % fit.n_features_)

print(("Selected Features: %s") % fit.support_)

print(("Feature Ranking: %s") % fit.ranking_)

features = []

for i , j in zip(X.columns,fit.support_):

    if j == True:

        features.append(str(i))
print(features)
from sklearn.model_selection import cross_val_score

import timeit

from xgboost import XGBClassifier

from statistics import mean

train_csv = pd.read_csv('../input/train.csv')
start = timeit.default_timer()

clf=XGBClassifier(base_score=0.5, colsample_bylevel=1, colsample_bytree=.8,subsample=0.5,

       gamma=0, learning_rate=0.1, max_delta_step=0, max_depth=5,

       missing=None, n_estimators=100, nthread=2,

       objective='multi:softprob', reg_alpha=0, reg_lambda=1,

       scale_pos_weight=1, seed=27, silent=True)

scores = cross_val_score(clf,X[features], label, cv=2)

print("Time take %.2f "%(timeit.default_timer()-start))

print("Accuracy: %0.4f (+/- %0.4f)" % (scores.mean(), scores.std() * 2))
xg = XGBClassifier(base_score=0.5, colsample_bylevel=1, colsample_bytree=.8,subsample=.2,

       gamma=0,learning_rate=0.1,max_delta_step= 4,max_depth=5,

       missing=None,n_estimators= 400,nthread=2,

       objective='multi:softprob', reg_alpha=0, reg_lambda=1,

       scale_pos_weight=1, seed=27,silent=1)

start = timeit.default_timer()

scores = cross_val_score(xg,X[features], label, cv=2)

print("Time take %.2f "%(timeit.default_timer()-start))

print("Accuracy: %0.4f (+/- %0.4f)" % (scores.mean(), scores.std() * 2))
start = timeit.default_timer()

xg.fit(X[features], label)

print("Time take to fit the data %.2f "%(timeit.default_timer()-start))
start = timeit.default_timer()

pre = xg.predict(test[features])

print("Time take predict output %.2f "%(timeit.default_timer()-start))