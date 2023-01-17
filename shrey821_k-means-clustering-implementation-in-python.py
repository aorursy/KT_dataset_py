import numpy as np

from sklearn.ensemble import AdaBoostClassifier

from sklearn.model_selection import train_test_split

from sklearn import metrics

from sklearn.model_selection import train_test_split 

from sklearn.linear_model import LogisticRegression 

from sklearn.neighbors import KNeighborsClassifier 

import pandas as pd

from sklearn.metrics import roc_curve, auc

from sklearn import datasets

from sklearn.multiclass import OneVsRestClassifier

from sklearn.svm import LinearSVC

from sklearn.preprocessing import label_binarize

import matplotlib.pyplot as plt

import sklearn.metrics

from matplotlib import pyplot as plt

from sklearn.metrics import confusion_matrix

from sklearn.metrics import classification_report

import matplotlib.colors as colors

import seaborn as sns

import itertools

from mlxtend.evaluate import lift_score

from scipy.stats import norm

import scipy.stats

from sklearn.model_selection import train_test_split

from sklearn.naive_bayes import GaussianNB

from sklearn.model_selection import KFold

from sklearn.metrics import confusion_matrix

from sklearn.metrics import classification_report

from sklearn import metrics

from sklearn.datasets import make_classification

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

from sklearn.metrics import roc_curve

from sklearn.metrics import roc_auc_score

from matplotlib import pyplot

from sklearn.metrics import confusion_matrix

from sklearn.metrics import classification_report

import os

import csv

import random

import math

from numpy import array

import numpy as np

from sklearn.model_selection import KFold

from sklearn.metrics import confusion_matrix

from sklearn.metrics import classification_report

from sklearn.datasets import make_classification

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

from sklearn.metrics import roc_curve

from sklearn.metrics import roc_auc_score

from matplotlib import pyplot

from sklearn.model_selection import KFold

from sklearn.metrics import confusion_matrix

from sklearn.metrics import classification_report

# def splitDataset(dataset, splitRatio=0.75):

#     trainSize = int(len(dataset) * splitRatio)

#     trainSet = []

#     copy = list(dataset)

#     while len(trainSet) < trainSize:

#         index = random.randrange(len(copy))

#         trainSet.append(copy.pop(index))

#     return [trainSet, copy]
center_1 = np.array([1,1])

center_2 = np.array([5,5])

center_3 = np.array([8,1])

center_4 = np.array([10,2])

data_1 = np.random.randn(200, 2) + center_1

data_2 = np.random.randn(200,2) + center_2

data_3 = np.random.randn(200,2) + center_3

data_4 = np.random.randn(200,2) + center_4

data = np.concatenate((data_1, data_2, data_3), axis = 0)

plt.scatter(data[:,0], data[:,1], s=7)
k = 4

n = data.shape[0]

c = data.shape[1]

mean = np.mean(data, axis = 0)

std = np.std(data, axis = 0)

centers = np.random.randn(k,c)*std + mean

plt.scatter(data[:,0], data[:,1], s=7)

plt.scatter(centers[:,0], centers[:,1], marker='*', c='g', s=150)
centers_old = np.zeros(centers.shape)

centers_new = deepcopy(centers) 

data.shape

clusters = np.zeros(n)

distances = np.zeros((n,k))

error = np.linalg.norm(centers_new - centers_old)

while error != 0:

    for i in range(k):

        distances[:,i] = np.linalg.norm(data - centers[i], axis=1)

    clusters = np.argmin(distances, axis = 1)    

    centers_old = deepcopy(centers_new)

    for i in range(k):

        centers_new[i] = np.mean(data[clusters == i], axis=0)

    error = np.linalg.norm(centers_new - centers_old)

centers_new    
plt.scatter(data[:,0], data[:,1], s=7)

plt.scatter(centers_new[:,0], centers_new[:,1], marker='*', c='g', s=150)
df = pd.read_csv("../input/Iris.csv")

df.drop('Id',axis=1,inplace=True)
df.head()
df["Species"] = pd.Categorical(df["Species"])

df["Species"] = df["Species"].cat.codes

data = df.values[:, 0:4]

category = df.values[:, 4]
k = 4 # number of cluster wanted

n = data.shape[0]

c = data.shape[1]

mean = np.mean(data, axis = 0)

std = np.std(data, axis = 0)

centers = np.random.randn(k,c)*std + mean

colors=['orange', 'blue', 'green']

for i in range(n):

    plt.scatter(data[i, 0], data[i,1], s=7, color = colors[int(category[i])])

plt.scatter(centers[:,0], centers[:,1], marker='*', c='g', s=150)
# with libary
import numpy as np

import matplotlib.pyplot as plt

import pandas as pd



#importing the Iris dataset with pandas

dataset = pd.read_csv('../input/Iris.csv')

x = dataset.iloc[:, [1, 2, 3, 4]].values
from sklearn.cluster import KMeans

wcss = []

for i in range(1, 11):

    kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)

    kmeans.fit(x)

    wcss.append(kmeans.inertia_)



plt.plot(range(1, 11), wcss)

plt.title('The elbow method')

plt.xlabel('Number of clusters')

plt.ylabel('WCSS')

plt.show()

kmeans = KMeans(n_clusters = 4, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)

y_kmeans = kmeans.fit_predict(x)
plt.scatter(x[y_kmeans == 0, 0], x[y_kmeans == 0, 1], c = 'red', label = 'setosa')

plt.scatter(x[y_kmeans == 1, 0], x[y_kmeans == 1, 1], c = 'blue', label = 'versicolour')

plt.scatter(x[y_kmeans == 2, 0], x[y_kmeans == 2, 1], c = 'green', label = 'virginica')



plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:,1], s = 90, c = 'purple', label = 'cluster center')



plt.legend()