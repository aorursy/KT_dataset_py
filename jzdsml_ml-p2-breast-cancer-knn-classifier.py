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
from sklearn.datasets import load_breast_cancer 



breast_cancer_data = load_breast_cancer()



X = breast_cancer_data.data

#check if need to reshape

print(X.shape)

X[0]
breast_cancer_data.feature_names
y = breast_cancer_data.target

print(y.shape)

y
#By looking at the target_names, we know that 0 corresponds to malignant..



breast_cancer_data.target_names
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 100)



#check splitting works properly

print(len(X_train)/len(X))

print(len(y_train)/len(y))
from sklearn.neighbors import KNeighborsClassifier

classifier = KNeighborsClassifier(n_neighbors = 3)

classifier.fit(X_train, y_train)

print(classifier.score(X_test, y_test))
k_score = []

k_list = range(1,101)

for k in k_list:

    classifier = KNeighborsClassifier(n_neighbors = k)

    classifier.fit(X_train, y_train)

    k_score.append(classifier.score(X_test, y_test))



from matplotlib import pyplot as plt

plt.figure(figsize=(12,7))

plt.plot(k_list, k_score)

plt.xlabel("k: numeber of neighbors")

plt.ylabel("score: validation accuracy")

plt.title("breast cancer knn classifier accuracy")

plt.show()

    