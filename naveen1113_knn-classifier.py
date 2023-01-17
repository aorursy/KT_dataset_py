# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

diabetes = pd.read_csv('/kaggle/input/dataset/diabetes.csv')

print(diabetes.columns)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(diabetes.loc[:, diabetes.columns != 'Outcome'], diabetes['Outcome'], stratify=diabetes['Outcome'], random_state=66)

from sklearn.neighbors import KNeighborsClassifier

training_accuracy = []

test_accuracy = []

# try n_neighbors from 1 to 10

neighbors_settings = range(1, 16)

for n_neighbors in neighbors_settings:

    # build the model

    knn = KNeighborsClassifier(n_neighbors=n_neighbors)

    knn.fit(X_train, y_train)

    # record training set accuracy

    training_accuracy.append(knn.score(X_train, y_train))

    # record test set accuracy

    test_accuracy.append(knn.score(X_test, y_test))

plt.plot(neighbors_settings, training_accuracy, label="training accuracy")

plt.plot(neighbors_settings, test_accuracy, label="test accuracy")

plt.ylabel("Accuracy")

plt.xlabel("n_neighbors")

plt.legend()

plt.savefig('knn_compare_model')

print("Best score is {} with K = {}".format(np.max(test_accuracy),1+test_accuracy.index(np.max(test_accuracy))))