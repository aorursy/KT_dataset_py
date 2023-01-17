# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# 'sep' is a regex because more than one row of the data has multiple tab separators so
# I use regex for flexible parsing.
# To perform regex with read_csv, the parsing engine has to be python 
# rather than the default C
seed_attr = ['area', 'perimeter', 'compactness', 'length', 'width', 
            'asymmetry_coef', 'groove_length']

seed_label = ['label']

seeds = pd.read_csv("../input/kama-rosa-and-canadian-seeds.txt", 
                    sep='\t+', 
                    engine='python',
                    header=None,
                    names = seed_attr + seed_label,
                   )
# display the first rows of data
seeds.head()
# Quick check for explicit nan values. 
seeds.info()
# since there isn't a test set separate from the given data, 
# let's split the data intp training and test sets. The transformations and models 
# that we decide to perform on the training data will need to generalize over the test set and
# future unknown data and not overfit to the data in hand
from sklearn.model_selection import train_test_split
# split 'seeds' DataFrame into attributes and labels
attributes = seeds.drop('label', axis=1)
labels = seeds.label

attributes_train, attributes_test, labels_train, labels_test = train_test_split(attributes, 
                                                                            labels, 
                                                                            test_size=0.2)
# Quick calculation of the range of values. We expect 'label' to be either 1, 2 or 3 from
# the question. The others will be continuous values.
attributes_train.describe().T
# check if the labels are skewed or more or less uniformly distributed.
labels_train.value_counts()
# visualize if there are attribute outliers using a box plot. An instance above or below
# the whiskers of the box plot can be considered outliers (Tukey's method). 
fig = plt.figure(figsize=(13, 7))
ax = attributes_train.boxplot()
ax.set(ylim=(0, 22.5));
# the attribute 'compactness' is about 1 order of magnitude smaller that the other attributes,
# so let's plot it again individually
fig = plt.figure(figsize=(10, 7))
ax = plt.axes()
ax.boxplot(attributes_train.compactness)
ax.set(xlim=(0.8, 1.2), xlabel=("Compactness"), 
       ylim=(0.80, 0.93));
# lets visualize the correlation between attributes
from pandas.plotting import scatter_matrix

ax = scatter_matrix(attributes, alpha=0.2, figsize=(13, 10), diagonal='kde')
attributes.corr()
# since there is high correlation between attributes, let's see if we can use 
# dimensionality reduction
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
lda = LinearDiscriminantAnalysis(n_components=2)
X_lda = lda.fit(attributes_train, labels_train).transform(attributes_train)

colors = ['navy', 'turquoise', 'darkorange']

for color, i, target_name in zip(colors, [1, 2, 3], ['kama', 'rosa', 'canadian']):
    plt.scatter(X_lda[labels_train == i, 0], X_lda[labels_train == i, 1], alpha=.8, color=color,
                label=target_name)
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.title('LDA of Kama, Rosa and Canadian seeds')
plt.xlabel("First Principal Component")
plt.ylabel("Second Principal Component");
# predict labels of the test set (not training set)
y_pred = lda.predict(attributes_test)
# transform the test set data into the 2 principal components for visualiation. 
# Note that we do not fit the test data again.
X_lda_test = lda.transform(attributes_test)

fig = plt.figure(figsize=(14, 7))
for color, i, train_labels, pred_labels, actual_labels in zip(colors, 
                                 [1, 2, 3], 
                                 ['kama (train)', 'rosa (train)', 'canadian (train)'], 
                                 ['kama (predicted)', 'rosa (predicted)', 'canadian (predicted)'], 
                                 ['kama (actual)', 'rosa (actual)', 'canadian (actual)']):
    plt.scatter(X_lda[labels_train == i, 0], X_lda[labels_train == i, 1], alpha=.8, color=color,
                label=train_labels)
    plt.scatter(X_lda_test[y_pred == i, 0], X_lda_test[y_pred == i, 1], marker='+', alpha=.8, color=color,
                label=pred_labels)
    plt.scatter(X_lda_test[labels_test == i, 0], X_lda_test[labels_test == i, 1], facecolor='none', 
                marker='o', s=100, edgecolor=color, alpha=.8, color=color,
                label=actual_labels)
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.title('LDA predicted labels of Kama, Rosa and Canadian seeds')
plt.xlabel("First Principal Component")
plt.ylabel("Second Principal Component");
# output the achieved accuracy 
from sklearn.metrics import accuracy_score
accuracy_score(labels_test, y_pred)
# output the metric of a naive method
accuracy_score(labels_test, np.repeat(1, len(labels_test)))
import pickle

filename = 'kama_rosa_canadian_seeds_LDA.pkl'
pickle.dump(lda, open(filename, 'wb'))