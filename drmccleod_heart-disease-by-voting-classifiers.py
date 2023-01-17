# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from speedml import Speedml

import sklearn

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

from sklearn import tree, svm

from sklearn.metrics import accuracy_score, auc, precision_recall_curve, f1_score, roc_auc_score, average_precision_score

from sklearn.metrics import recall_score, precision_score;

from sklearn.naive_bayes import GaussianNB

from sklearn.neural_network import MLPClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn import preprocessing

from scipy import stats

import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/heart.csv')

df
#normalise

x = df.values

min_max_scaler = preprocessing.MinMaxScaler()

x_scaled = min_max_scaler.fit_transform(x)

df = pd.DataFrame(x_scaled, columns=df.columns)

df
mask = np.random.rand(len(df)) < 2/3
train = df[mask]

test = df[~mask]



train.to_csv('train.csv')

test.to_csv('test.csv')
sml = Speedml('train.csv', 'test.csv', 'target')

sml.eda()
X = train.drop('target', axis=1)

y = train['target']
clf = RandomForestClassifier(n_estimators=400, max_depth=2, random_state=0)



clf.fit(train.drop('target', axis=1), train['target'])

random_forest_predictions = clf.predict(test.drop('target', axis=1))

accuracy_score(test['target'], random_forest_predictions)
clf = tree.DecisionTreeClassifier(criterion='gini', splitter='best')

clf.fit(train.drop('target', axis=1), train['target'])

decision_tree_predictions = clf.predict(test.drop('target', axis=1))

accuracy_score(test['target'], decision_tree_predictions)
clf = svm.SVC(gamma='scale', C=.5, kernel='linear')

clf.fit(train.drop('target', axis=1), train['target'])

svm_predictions = clf.predict(test.drop('target', axis=1))

accuracy_score(test['target'], svm_predictions)
clf = AdaBoostClassifier()

clf.fit(train.drop('target', axis=1), train['target'])

ada_predictions = clf.predict(test.drop('target', axis=1))

accuracy_score(test['target'], ada_predictions)
clf = GaussianNB()

clf.fit(train.drop('target', axis=1), train['target'])

gauss_predictions = clf.predict(test.drop('target', axis=1))

accuracy_score(test['target'], gauss_predictions)
clf = MLPClassifier(solver='lbfgs')

clf.fit(train.drop('target', axis=1), train['target'])

mlp_predictions = clf.predict(test.drop('target', axis=1))

accuracy_score(test['target'], mlp_predictions)
clf = KNeighborsClassifier(n_neighbors = 6)

clf.fit(train.drop('target', axis=1), train['target'])

kneighours_predictions=clf.predict(test.drop('target', axis=1))

accuracy_score(test['target'], kneighours_predictions)
vote = stats.mode([kneighours_predictions,mlp_predictions, gauss_predictions, ada_predictions, decision_tree_predictions, svm_predictions, random_forest_predictions])

vote = vote[0][0]

print("Accuracy: ", accuracy_score(test['target'], vote))

print("F1: ", f1_score(test['target'], vote))

print("Area under ROC curve: ", roc_auc_score(test['target'], vote))

print("Average precision", average_precision_score(test['target'], vote))

print("Precision: ", precision_score(test['target'], vote))

print("Recall: ", recall_score(test['target'], vote))


