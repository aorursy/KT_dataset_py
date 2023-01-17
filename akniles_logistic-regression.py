import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
import seaborn as sns
import warnings
from sklearn.model_selection import RandomizedSearchCV

pd.options.mode.chained_assignment = None  # default='warn'

with warnings.catch_warnings():
    warnings.simplefilter("ignore")

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
train = pd.read_csv("/kaggle/input/hsemath2020flights/flights_train.csv")
test = pd.read_csv("/kaggle/input/hsemath2020flights/flights_test.csv")
train.isnull().values.any(), test.isnull().values.any()
train.columns
target = 'dep_delayed_15min'
train.describe()
train[target].describe()
for feature in ('DEPARTURE_TIME', 'DISTANCE'):
    uniques = sorted(train[feature].unique())
    average_target = []
    for value in uniques:
        average_target.append(np.sum(train[target][train[feature] == value]) / len(train[target][train[feature] == value])) # доля задержанных рейсов от всех рейсов с этим значением признака
    plt.plot(uniques, average_target, label=feature)
    plt.legend(loc='best')
    plt.show()
for feature in ('DATE', 'AIRLINE', 'ORIGIN_AIRPORT','DESTINATION_AIRPORT'):
    print(feature +  ":")
    uniques = sorted(train[feature].unique())
    average_target = []
    for value in uniques:
        average_target.append(np.sum(train[target][train[feature] == value]) / len(train[target][train[feature] == value]))
    f, ax = plt.subplots(figsize=(8, 6))
    fig = sns.boxplot(x=uniques, y=average_target)
    fig.axis(ymin=0, ymax=1);
sorted(train['DATE'].unique())
set(train['DATE'].unique()).symmetric_difference(set(test['DATE'].unique()))
train, test = train.drop(['AIRLINE', 'ORIGIN_AIRPORT', 'DESTINATION_AIRPORT'], axis=1), test.drop(['AIRLINE', 'ORIGIN_AIRPORT', 'DESTINATION_AIRPORT'], axis=1)
train['DATE'] = train['DATE'].apply(lambda x: x.split('-')[1])
test['DATE'] = test['DATE'].apply(lambda x: x.split('-')[1])
train['DATE']
y, X = train['dep_delayed_15min'], train.drop('dep_delayed_15min', axis=1)
testsize = test.shape[0]
temp = test.append(X)
temp = pd.get_dummies(temp)
test = temp[:testsize]
X = temp[testsize:]
model = LogisticRegression()
cross_val_score(model, X, y, cv=4, scoring='roc_auc')
hyperparameter = {'penalty': ['l2'], 'C': np.linspace(0.1, 1, 10)}
search = RandomizedSearchCV(model, hyperparameter, n_iter=10)
search.fit(X, y)
l2 = search.best_params_
print(l2)
model = LogisticRegression(C=l2['C'])
cross_val_score(model, X, y, cv=4, scoring='roc_auc')
model.fit(X, y)
prediction = model.predict_proba(test)
sample_sub = pd.read_csv('/kaggle/input/hsemath2020flights/flights_sample_submission.csv', index_col='id')
sample_sub['dep_delayed_15min'] = prediction[:, 1]
sample_sub.to_csv('log_prediction.csv')