import warnings

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

from sklearn.preprocessing import StandardScaler

# Preprocessing :

from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import classification_report,confusion_matrix

from itertools import product

import seaborn as sns

import matplotlib.pyplot as plt



# Classifiers

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn import svm

from sklearn import tree

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.naive_bayes import GaussianNB

from sklearn.manifold import TSNE

from sklearn.decomposition import PCA

from xgboost import XGBClassifier

from xgboost import plot_tree

from sklearn.manifold import TSNE

import time

from pandas import datetime

from statsmodels.tsa.arima_model import ARIMA



#Cloustering

import scipy.cluster.hierarchy as sch

from sklearn.cluster import AgglomerativeClustering

from sklearn.cluster import KMeans
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
warnings.filterwarnings('ignore')
train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")
test.head()
#X_Train = train.drop('target', axis=1)

X_Test = test.drop('id', axis=1).values

#Y_Train = train['target']

X_Test = StandardScaler().fit_transform(X_Test)
train.head()

#print(train.shape)
X_Train = train.drop('target', axis=1)

X_Train = X_Train.drop('id', axis=1).values

Y_Train = train['target']

X_Train = StandardScaler().fit_transform(X_Train)
plt.figure(figsize = (20,15))

corr=test.corr()

sns.heatmap(corr, xticklabels=corr.columns.values, yticklabels=corr.columns.values)
plt.figure(figsize = (20,15))

corr=train.corr()

sns.heatmap(corr, xticklabels=corr.columns.values, yticklabels=corr.columns.values)
trainedmodel = LogisticRegression().fit(X_Train,Y_Train)

predictions =trainedmodel.predict(X_Test)
print(predictions)
trainedtree = tree.DecisionTreeClassifier().fit(X_Train, Y_Train)

predictionstree = trainedtree.predict(X_Test)
import graphviz

from sklearn.tree import DecisionTreeClassifier, export_graphviz



data = export_graphviz(trainedtree,out_file=None,feature_names=test.drop(['id'], axis = 1).columns,

                       class_names=['0', '1'],  

                       filled=True, rounded=True,  

                       max_depth=2,

                       special_characters=True)

graph = graphviz.Source(data)

graph
treepre = predictions

print(treepre)
submission = pd.read_csv('../input/sample_submission.csv')

submission['target'] = treepre

submission.to_csv('submission.csv', index=False)
model = XGBClassifier()



# Train

model.fit(X_Train, Y_Train)



plot_tree(model)

plt.figure(figsize = (50,55))

plt.show()
pca = PCA(n_components=2,svd_solver='full')

X_reduced = pca.fit_transform(X_Train)

X_test_reduced = pca.fit_transform(X_Test)
reduced_data = X_reduced



trainednb = GaussianNB().fit(reduced_data, Y_Train)

trainedsvm = svm.LinearSVC().fit(reduced_data, Y_Train)

trainedforest = RandomForestClassifier(n_estimators=700).fit(reduced_data,Y_Train)

trainedmodel = LogisticRegression().fit(reduced_data,Y_Train)



# Thanks to: https://scikit-learn.org/stable/auto_examples/ensemble/plot_voting_decision_regions.html



x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1

y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1

xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),

                     np.arange(y_min, y_max, 0.1))



f, axarr = plt.subplots(2, 2, sharex='col', sharey='row', figsize=(10, 8))



for idx, clf, tt in zip(product([0, 1], [0, 1]),

                        [trainednb, trainedsvm, trainedforest, trainedmodel],

                        ['Naive Bayes Classifier', 'SVM',

                         'Random Forest', 'Logistic Regression']):



    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    Z = Z.reshape(xx.shape)

    



    axarr[idx[0], idx[1]].contourf(xx, yy, Z,cmap=plt.cm.coolwarm, alpha=0.4)

    axarr[idx[0], idx[1]].scatter(reduced_data[:, 0], reduced_data[:, 1], c=Y_Train,

                                  s=20, edgecolor='k')

    axarr[idx[0], idx[1]].set_title(tt)



plt.show()
kmeans = KMeans(n_clusters=2, random_state=0).fit(X_reduced)

kpredictions = kmeans.predict(X_test_reduced)
plt.scatter(X_test_reduced[kpredictions ==0,0], X_test_reduced[kpredictions == 0,1], s=100, c='red')

plt.scatter(X_test_reduced[kpredictions ==1,0], X_test_reduced[kpredictions == 1,1], s=100, c='black')