# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))

train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")

targets = train['target']

train = train.drop(columns=['id', 'target'])

ids = test['id']

test = test.drop(columns=['id'])
from sklearn.preprocessing import RobustScaler

transformer = RobustScaler().fit(test.values)

trans_data = transformer.transform(train.values)

trans_test = transformer.transform(test.values)



trans_data_df = pd.DataFrame(trans_data, columns=train.columns)

trans_test_df = pd.DataFrame(trans_test, columns=test.columns)
from sklearn.cluster import KMeans, DBSCAN, MeanShift

from sklearn.mixture import GaussianMixture

center_point_0 = trans_data_df[targets == 0].mean()

#print(center_point_0)

center_point_1 = trans_data_df[targets == 1].mean()

start = np.array([center_point_0, center_point_1])

kmeans = KMeans(n_clusters=2, init=start).fit(test.values)

g_mix = GaussianMixture(n_components=2).fit(test.values)

#db_scan = DBSCAN(eps=75, min_samples=10).fit(test.values)

predicts = kmeans.predict(train.values)

g_predicts= g_mix.predict(train.values)

count= 0

for i in range(len(predicts)):

    if targets[i] == predicts[i]:

        count+=1

print("K-Means Cluster Matches Target : ", str(count), " times out of 250")

data = pd.DataFrame(trans_data.copy(), columns=train.columns)

data['k_cluster'] = predicts

data['g_mix'] = g_predicts

t_data = pd.DataFrame(trans_test.copy(), columns=test.columns)

t_data['k_cluster'] = kmeans.predict(test.values)

t_data['g_mix'] = g_mix.predict(test.values)
count=0

for i in range(len(g_predicts)):

    if targets[i] == g_predicts[i]:

        count+=1

print("For Gaussian-Mixture : ", count)
from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import cross_validate

from sklearn.model_selection import GridSearchCV

import matplotlib.pyplot as plt

import heapq

print(data.head())

rf = RandomForestClassifier(n_estimators=300, max_depth=5,random_state=0)

rf.fit(data.values, targets)

my_heap = []

for i in range(len(rf.feature_importances_)):

    heapq.heappush(my_heap, (-rf.feature_importances_[i], data.columns[i]))

heights = []

labels = []

for i in range(10):

    height, label = heapq.heappop(my_heap)

    heights.append(-height)

    labels.append(label)

plt.bar(range(10), height=heights, tick_label=labels)



plt.show()
param_grid = {"C": [.01, .1, 1, 10, 100, 1000],

                "penalty": ('l1','l2')}

top_data = data.drop(columns=['k_cluster', 'g_mix'])

clf = LogisticRegression(random_state=0, solver='liblinear', class_weight='balanced', max_iter = 1000)

gs = GridSearchCV(clf, param_grid, cv=5)

gs.fit(top_data.values, targets)

print(gs.best_score_)

print(gs.best_estimator_)

predictions = gs.predict_proba(t_data.drop(columns=['k_cluster', 'g_mix']).values)[:,1]

output = pd.DataFrame({'id' : ids, 'target' : predictions})

output.to_csv('scale_only_output.csv',index=None)
param_grid = {"C": [.01, .1, 1, 10, 100, 1000],

                "penalty": ('l1','l2')}

top_data = data

clf = LogisticRegression(random_state=0, solver='liblinear', class_weight='balanced', max_iter = 1000)

gs = GridSearchCV(clf, param_grid, cv=5)

gs.fit(top_data.values, targets)

print(gs.best_score_)

print(gs.best_estimator_)

predictions = gs.predict_proba(t_data.values)[:,1]

output = pd.DataFrame({'id' : ids, 'target' : predictions})

output.to_csv('scale_and_clustering_output.csv',index=None)
param_grid = {"C": [.01, .1, 1, 10, 100, 1000],

                "penalty": ('l1','l2')}

top_data = data[labels]

clf = LogisticRegression(random_state=0, solver='liblinear', class_weight='balanced', max_iter = 1000)

gs = GridSearchCV(clf, param_grid, cv=5)

gs.fit(top_data.values, targets)

print(gs.best_score_)

print(gs.best_estimator_)

predictions = gs.predict_proba(t_data[labels].values)[:,1]

output = pd.DataFrame({'id' : ids, 'target' : predictions})

output.to_csv('reduced_scale_and_clustering_output.csv',index=None)