# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
#Reading data from CSV file

df = pd.read_csv("../input/Iris.csv")
#Defining data and label

X = df.iloc[:, 1:5]

y = df.iloc[:, 5]
#importing some resources for preprocessing, creating pipeline, creating splits from dataset and CV

from sklearn.pipeline import make_pipeline

from sklearn import preprocessing

from sklearn.model_selection import  ShuffleSplit

from sklearn.model_selection import cross_val_score
#Applying a arbitraty Knn model

from sklearn.neighbors import KNeighborsClassifier



cv = ShuffleSplit(n_splits=5, test_size=0.3, random_state=0)

clf_knn = make_pipeline(preprocessing.StandardScaler(), KNeighborsClassifier(n_neighbors = 7, p = 2, metric='minkowski'))

scores_knn = cross_val_score(clf_knn, X, y, cv = cv)



print("Accuracy of Knn: %0.3f (+/- %0.2f)" % (scores_knn.mean(), scores_knn.std()))
from sklearn import metrics



#initialize vector for storing accuracies

v_1 = []

v_2 = []

v_5 = []



#set range of k to be analized

k_range = list(range(1, 50))



#loop over k, using three different distance methods

for i in k_range:

    clf_knn_1 = make_pipeline(preprocessing.StandardScaler(), KNeighborsClassifier(n_neighbors = i, p = 1))

    scores_knn_1 = cross_val_score(clf_knn_1, X, y, cv = cv)

    v_1.append(scores_knn_1.mean())

    clf_knn_2 = make_pipeline(preprocessing.StandardScaler(), KNeighborsClassifier(n_neighbors = i, p = 2))

    scores_knn_2 = cross_val_score(clf_knn_2, X, y, cv = cv)

    v_2.append(scores_knn_2.mean())

    clf_knn_5 = make_pipeline(preprocessing.StandardScaler(), KNeighborsClassifier(n_neighbors = i, p = 3))

    scores_knn_5 = cross_val_score(clf_knn_5, X, y, cv = cv)

    v_5.append(scores_knn_5.mean())

    

#plotting accuracy 

plt.figure(figsize=(10, 10))

plt.subplot(2, 1, 1)

plt.plot(k_range, v_1, c = 'r', label = 'Manhattan distance')

plt.plot(k_range, v_2, c = 'g', label = 'Euclidian distance')

plt.plot(k_range, v_5, c = 'y', label = 'Minkowski distance')

plt.legend()

plt.title('Knn Accuracy vs k value by distance method')

plt.ylabel('Accuracy')

plt.xlabel('k')

plt.show()



#initialize vector for storing accuracies

v_3 = []

v_4 = []



#loop over k, using three different weight methods

for i in k_range:

    clf_knn_3 = make_pipeline(preprocessing.StandardScaler(), KNeighborsClassifier(n_neighbors = i, weights='uniform'))

    scores_knn_3 = cross_val_score(clf_knn_3, X, y, cv = cv)

    v_3.append(scores_knn_3.mean())

    clf_knn_4 = make_pipeline(preprocessing.StandardScaler(), KNeighborsClassifier(n_neighbors = i, weights='distance'))

    scores_knn_4 = cross_val_score(clf_knn_4, X, y, cv = cv)

    v_4.append(scores_knn_4.mean())

    

#plotting accuracy

plt.figure(figsize=(10, 10))

plt.subplot(2, 1, 2)

plt.plot(k_range, v_3, c = 'b', label = 'Uniform weight')

plt.plot(k_range, v_4, c = 'y', label = 'Distance weight')

plt.legend()

plt.title('Knn Accuracy vs k value by weight method')

plt.ylabel('Accuracy')

plt.xlabel('k')

plt.show()