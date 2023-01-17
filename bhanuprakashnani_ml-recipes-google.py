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
from scipy.spatial import distance



def euc(a,b):

    return distance.euclidean(a,b)
import random



class ScrappyKNN():

    def fit(self, X_train, y_train):

        self.X_train = X_train

        self.y_train = y_train

    

    def predict(self, X_test):

        predictions = []

        for row in X_test:

            label = self.closest(row)

            predictions.append(label)

        

        return predictions

    

    def closest(self, row):

        best_dist = euc(row, self.X_train[0])

        best_index = 0

        for i in range(1, len(self.X_train)):

            dist = euc(row, self.X_train[i])

            if(dist < best_dist):

                best_dist = dist

                best_index = i

        return self.y_train[best_index]

                

    

    
from sklearn import datasets

iris = datasets.load_iris()



X = iris.data

y = iris.target



from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)
from sklearn.neighbors import KNeighborsClassifier

my_classifier = ScrappyKNN()

my_classifier.fit(X_train, y_train)
predictions = my_classifier.predict(X_test)
from sklearn.metrics import accuracy_score

print(accuracy_score(y_test, predictions))