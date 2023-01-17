import seaborn as sns

import pandas as pd, numpy as np, textblob as tb, string as stri

import math

import csv,re

import time

import os

import pylab as pl

import random

from sklearn import model_selection, preprocessing, metrics

from sklearn.preprocessing import StandardScaler

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

from sklearn.naive_bayes import GaussianNB as GNB, MultinomialNB as MNB, BernoulliNB as BNB

from sklearn.tree import DecisionTreeClassifier as DTC

from sklearn.decomposition import PCA

from sklearn.mixture import GaussianMixture as GMM

from sklearn.neighbors import KNeighborsClassifier

from sklearn import datasets, svm, metrics

from sklearn.neighbors import KernelDensity

from sklearn import tree

from sklearn.model_selection import train_test_split

from sklearn.model_selection import GridSearchCV

from sklearn.impute import KNNImputer

import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression

le = preprocessing.LabelEncoder()       
D0 = pd.read_csv('/kaggle/input/digit-recognizer/train.csv', sep=',', encoding='latin-1')

D0.head()
D1 = D0.drop('label', axis = 1)

D1.head()
pca_f = PCA(n_components = 9)

D2 = pd.DataFrame(pca_f.fit_transform(D1.drop(D1.std()[D1.std() == 0.0].index.values, axis = 1)),  columns = ["PC"+str(i) for i in range(1,10)])

D2.head()
D2['Label'] = D0['label']

D2['Label'].unique()
test = pd.read_csv('/kaggle/input/digit-recognizer/test.csv', sep=',', encoding='latin-1')

test.head()
test_pca = pd.DataFrame(pca_f.fit_transform(test.drop(test.std()[test.std() == 0.0].index.values, axis = 1)),  columns = ["PC"+str(i) for i in range(1,10)])

test_pca.head()
D1_x_KNN = D2.iloc[:, 0:-1].values

D1_y_KNN = D2.iloc[:, -1].values



trainset_x, valset_x, trainset_y, valset_y = train_test_split(D1_x_KNN,D1_y_KNN, test_size=0.3, random_state = 2)
column3 = np.arange(1,13,2)

df3 = pd.DataFrame(column3, columns = ["K"])

train_accuracy = []

val_accuracy = []

for k in column3:

    KNN = KNeighborsClassifier(n_neighbors=k)

    test_fit_knn = KNN.fit(trainset_x, trainset_y)

    val_pred_knn = test_fit_knn.predict(valset_x)

    train_pred_knn = test_fit_knn.predict(trainset_x)

    val_accuracy.append(metrics.accuracy_score(valset_y, val_pred_knn))

    train_accuracy.append(metrics.accuracy_score(trainset_y, train_pred_knn))

df3['train_accuracy'] = train_accuracy

df3['val_accuracy'] = val_accuracy

df3
plt.plot('K', 'train_accuracy', data=df3, marker='', color='skyblue', linewidth=4)

plt.plot('K', 'val_accuracy', data=df3, marker='', color='olive', linewidth=4)

plt.legend()
print('KNN Classifier + PCA:')

df3[df3['val_accuracy'] == df3['val_accuracy'].max()].iloc[[-1]]
KNN = KNeighborsClassifier(n_neighbors=9)

test_fit_knn = KNN.fit(trainset_x, trainset_y)

test_pred_knn = test_fit_knn.predict(test_pca)
column4 = np.arange(1,28001,1)

submit = pd.DataFrame(column4, columns = ["ImageId"])

submit['Label'] = pd.DataFrame(test_pred_knn)

submit.tail()
os.chdir(r'/kaggle/working')

from IPython.display import FileLink

submit.to_csv(r'mnist_pca+knn.csv', index = False, header=True)