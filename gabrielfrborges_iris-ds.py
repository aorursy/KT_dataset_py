# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from plotnine import *

import seaborn as sns

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# read the data :D

iris = pd.read_csv('../input/iris-flower-dataset/IRIS.csv')
#check data first lines

iris.head()
# statistical summary

iris.describe()
# visualize data in a matrix scatter plot

sns.pairplot(data = iris, hue = 'species', 

             hue_order  = ['Iris-setosa', 'Iris-virginica', 'Iris-versicolor'])

plt.show()
# distribution of attributes

fig, ax = plt.subplots(figsize=(8,6))

sns.boxplot(data = iris)

plt.show()
# calculate correation of variables

corr = iris.corr()

corr
# plot correlation heatmap

ax = sns.heatmap(corr, annot = True, linewidths = 1)

ax.set_title('Correlation Matrix\n', 

            fontdict  = {'fontweight':'bold'})

plt.show()
from sklearn.model_selection import train_test_split
# create train/test dataframes

train, test = train_test_split(iris, test_size=0.33, random_state = 7)

# reset index

train.reset_index(inplace = True)

test.reset_index(inplace = True)
x_test = test.loc[:, 'sepal_length':'petal_width']

y_test = test['species']
x_train = train.loc[:, 'sepal_length':'petal_width']

y_train = train['species']
sns.pairplot(train, hue = 'species',  hue_order  = ['Iris-setosa', 'Iris-virginica', 'Iris-versicolor'])

plt.show()
sns.pairplot(test, hue = 'species', hue_order  = ['Iris-setosa', 'Iris-virginica', 'Iris-versicolor'])

plt.show()
plt.figure(figsize = (12,8))

cmap = sns.cubehelix_palette(dark=.3, light=.8, as_cmap=True)

sns.scatterplot(data = iris, x = 'petal_length', y = 'petal_width', 

                size = 'sepal_length', hue = 'species')

plt.show()
# import knn classifier and metrics

from sklearn.neighbors import KNeighborsClassifier

from sklearn import metrics
metric = {'k':[], 'acc':[]}



for n in range(1,6):

    knn = KNeighborsClassifier(n_neighbors =  n)

    knn.fit(x_train, y_train)

    y_pred = knn.predict(x_test)

    acc = metrics.accuracy_score(y_test, y_pred)

    metric['k'].append(n)

    metric['acc'].append(acc)

metric = pd.DataFrame(metric)

metric
metric = {'k':[], 'acc':[]}

for n in range(1,6):

    knn = KNeighborsClassifier(n_neighbors =  n)

    knn.fit(x_train, y_train)

    y_pred = knn.predict(x_test)

    acc = metrics.accuracy_score(y_test, y_pred)

    metric['k'].append(n)

    metric['acc'].append(acc)

metric = pd.DataFrame(metric)

metric
# Tree

from sklearn import tree

from sklearn.tree import DecisionTreeClassifier
metric = {'n':[], 'acc':[]}

for n in range(2, 5):

    clf =  DecisionTreeClassifier(max_depth = n)

    clf.fit(x_train, y_train)

    y_pred = clf.predict(x_test)

    acc = metrics.accuracy_score(y_test, y_pred)

    metric['n'].append(n)

    metric['acc'].append(acc)

metric = pd.DataFrame(metric)

metric
import graphviz
# >>> dot_data = tree.export_graphviz(clf, out_file=None, 

# ...                      feature_names=iris.feature_names,  

# ...                      class_names=iris.target_names,  

# ...                      filled=True, rounded=True,  

# ...                      special_characters=True)  

# >>> graph = graphviz.Source(dot_data)  

# >>> graph 

f_names = x_test.columns

c_names = np.unique(y_test)

dot_data = tree.export_graphviz(clf, out_file=None, 

                                feature_names = f_names, class_names = c_names,

                               filled = True, rounded = True,

                               special_characters = True)

graph = graphviz.Source(dot_data)

graph
cm = metrics.confusion_matrix(y_test, y_pred, c_names)

cm = cm / cm.sum()

ax = sns.heatmap(cm, annot = True, linewidth = .5, cmap = 'viridis')

ax.set_xticklabels(c_names)

ax.set_yticklabels(c_names)

plt.yticks(rotation = 0)

plt.show()