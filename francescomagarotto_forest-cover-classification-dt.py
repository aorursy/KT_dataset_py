# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import pylab as plt

from sklearn import tree

from sklearn.tree import export_graphviz

from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import train_test_split

from sklearn.metrics import precision_score, accuracy_score, recall_score

import graphviz

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('/kaggle/input/forest-cover-type-dataset/covtype.csv')

x_names = set(data.columns) - set(f'Soil_Type{i}' for i in range(1, 40))- {'Cover_Type'}
attributes = list(x_names)

X = data[attributes]

scaler = MinMaxScaler()

X_scaled = X.copy()

scaler.fit(X)

X_scaled = scaler.transform(X)

y = data['Cover_Type']
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.33, random_state=0)
accuracy_scores = list()

average_precision_scores = list()

recall = list()

depths = range(3, 7)

trees = list()

for depth in depths:

    clf = tree.DecisionTreeClassifier(max_depth=depth, criterion='entropy')

    #alleno il modello

    clf = clf.fit(X_train, y_train)

    #faccio predizione sul test set

    y_pred = clf.predict(X_test)

    average_precision_scores.append(precision_score(y_test, y_pred, average='micro'))

    accuracy_scores.append(accuracy_score(y_test, y_pred))

    recall.append(recall_score(y_test, y_pred, average='micro'))

    trees.append(clf)

acc_data = pd.DataFrame({"Accuracy":accuracy_scores, "Average precision score": average_precision_scores, "Recall": recall, "Depth": list(depths)})

acc_data
plt.style.use('ggplot')

fig = plt.figure(1)

ax = plt.gca()

ax.plot(depths, accuracy_scores, lw = 1)

plt.draw()
dot_data = tree.export_graphviz(trees[0], max_depth=3, filled=True, rounded=True, feature_names=attributes)

graph = graphviz.Source(dot_data)

graph