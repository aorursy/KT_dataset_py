# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/fashion-mnist_train.csv', dtype=int)

test = pd.read_csv('../input/fashion-mnist_test.csv', dtype=int)

X_train = train.drop('label', axis=1)

y_train = train[['label']]

X_test = test.drop('label', axis=1)

y_test = test[['label']]
import warnings

warnings.filterwarnings('ignore')
def plot_matrix(m):

    fig = plt.figure(figsize=m.shape)

    ax = fig.add_subplot(111)

    cax = ax.matshow(m)

    fig.colorbar(cax)
import matplotlib.pyplot as plt

plot1 = X_train[0:1].values.reshape(28,28)

plot2 = X_train[1:2].values.reshape(28,28)

plot3 = X_train[2:3].values.reshape(28,28)

plot_matrix(plot1)

plot_matrix(plot2)

plot_matrix(plot3)
from sklearn.naive_bayes import GaussianNB

g_clf = GaussianNB()



g_clf.fit(X_train,y_train)

g_pred = g_clf.predict(X_test)

g_pred



from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, g_pred)

accuracy
from sklearn.ensemble import RandomForestClassifier

rf_clf = RandomForestClassifier()
def grid_search(CLF, X_train, y_train, num_estimators):

    highest_score = 0;

    for features in (1,4,16,64,'auto'):

        for depth in (1,4,16,64, None):

            CLF = RandomForestClassifier(max_features=features, max_depth=depth, n_estimators=num_estimators,oob_score=True, bootstrap=True, random_state=1105233, n_jobs=-1)

            CLF.fit(X_train, y_train)

            oob_score = CLF.oob_score_

            print(depth, features, oob_score)

        if(oob_score > highest_score):

            highest_score = oob_score

            max_depth = depth

            max_features = features

            result = [highest_score, max_depth, max_features]

    return result    
search = grid_search(rf_clf, X_train, y_train, 30)

print(search)
from sklearn.metrics import confusion_matrix

new_rf_clf = RandomForestClassifier(max_depth=None, max_features=64, n_estimators=30, n_jobs=-1, random_state=1105233)

new_rf_clf.fit(X_train, y_train)

rf_pred = new_rf_clf.predict(X_test)

rf_accuracy = accuracy_score(y_test, rf_pred)

rf_confusion = confusion_matrix(y_test, rf_pred)

print(rf_accuracy)

print(rf_confusion)
import seaborn as sns

plt.figure(figsize=(14,7))

sns.heatmap(rf_confusion, annot=True)

plt.title("Random Forest Confusion")
feat_importances = new_rf_clf.feature_importances_

feat_matrix = feat_importances.reshape(28,28)

plot_matrix(feat_matrix)
def ex_grid_search(CLF, X_train, y_train, num_estimators):

    highest_score = 0;

    for features in (1,4,16,64,'auto'):

        for depth in (1,4,16,64, None):

            CLF = ExtraTreesClassifier(max_features=features, max_depth=depth, n_estimators=num_estimators,oob_score=True, bootstrap=True, random_state=1105233, n_jobs=-1)

            CLF.fit(X_train, y_train)

            oob_score = CLF.oob_score_

            print(depth, features, oob_score)

        if(oob_score > highest_score):

            highest_score = oob_score

            max_depth = depth

            max_features = features

            result = [highest_score, max_depth, max_features]

    return result  
from sklearn.ensemble import ExtraTreesClassifier

ex_clf = ExtraTreesClassifier()

search = ex_grid_search(ex_clf, X_train, y_train, 100)

print(search)
ex_clf = ExtraTreesClassifier(max_features=64, n_estimators=100, n_jobs=-1, random_state=1105233)

ex_clf.fit(X_train, y_train)

ex_pred = ex_clf.predict(X_test)

ex_accuracy = accuracy_score(y_test, ex_pred)

ex_confusion = confusion_matrix(y_test, ex_pred)

print(ex_accuracy)

print(ex_confusion)
plt.figure(figsize=(14,7))

sns.heatmap(ex_confusion, annot=True)

plt.title("Extra Trees Confusion")
ex_feat_importances = ex_clf.feature_importances_

ex_feat_matrix = ex_feat_importances.reshape(28,28)

plot_matrix(ex_feat_matrix)