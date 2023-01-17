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
#reading data

data = pd.read_csv('../input/heart.csv')
#recognizing features

data.head()
#info about features and target

data.info()



#there is no NaN value or not needed feature in this dataset.
data.describe()



#the values of features are needed to be normalized.
#preperation of data:



#y is target column

#x_data is feature set without normalization

y = data.target.values

x_data = data.drop(['target'], axis = 1)
#normalization of feature set

x = (x_data - np.min(x_data))/(np.max(x_data)-np.min(x_data)).values
#train_test_split

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.20, random_state = 42)
#logistic regression classification model with sklearn

from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(random_state = 42, max_iter = 150)

lr.fit(x_train, y_train)
#accuracy of the model

lr.score(x_test, y_test)
#data scheme for confusion matrix

y_pred = lr.predict(x_test)

y_true = y_test
#confusion matrix model with sklearn

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_true, y_pred)

cm
#KNN model with sklearn

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors = 18) #n_neighbors = K

knn.fit(x_train, y_train)

#accuracy of the model

knn.score(x_test, y_test)
#KNN K-values evaluation results



score_list = []

for i in range(1,25):

    knn2 = KNeighborsClassifier(n_neighbors=i)

    knn2.fit(x_train, y_train)

    score_list.append(knn2.score(x_test, y_test))

plt.plot(range(1,25), score_list)

plt.xlabel('k_values')

plt.ylabel('accuracy')

plt.show()



#n_neighbors should be 18
#data scheme for confusion matrix

y_pred = knn.predict(x_test)

y_true = y_test

#confusion matrix model with sklearn

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_true, y_pred)

cm
#SVM model with sklearn

from sklearn.svm import SVC

svm = SVC(random_state = 1)

svm.fit(x_train, y_train)
#accuracy of the model

svm.score(x_test, y_test)
#data scheme for confusion matrix

y_pred = svm.predict(x_test)

y_true = y_test

#confusion matrix model with sklearn

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_true, y_pred)

cm
#NAIVE BAYES model with sklearn

from sklearn.naive_bayes import GaussianNB

nb = GaussianNB()

nb.fit(x_train, y_train)
#accuracy of the model

nb.score(x_test, y_test)
#data scheme for confusion matrix

y_pred = nb.predict(x_test)

y_true = y_test

#confusion matrix model with sklearn

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_true, y_pred)

cm
#Decision Tree Classification model with sklearn

from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier()

dt.fit(x_train, y_train)
#accuracy of the model

dt.score(x_test, y_test)
#data scheme for confusion matrix

y_pred = dt.predict(x_test)

y_true = y_test

#confusion matrix model with sklearn

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_true, y_pred)

cm
#Random Forest Classification model with sklearn

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators = 100, random_state = 1)

rf.fit(x_train, y_train)
#accuracy of the model

rf.score(x_test, y_test)
#data scheme for confusion matrix

y_pred = rf.predict(x_test)

y_true = y_test

#confusion matrix model with sklearn

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_true, y_pred)

cm
L = [lr.score(x_test, y_test),knn.score(x_test, y_test),svm.score(x_test, y_test),nb.score(x_test, y_test),dt.score(x_test, y_test),rf.score(x_test, y_test)]

print('ACCURACY SCORES OF THE MODELS')

print('Linear Regression      : ',L[0].round(3))

print('KNN                    : ',L[1].round(3))

print('Support Vector Machine : ',L[2].round(3))

print('Naive Bayes            : ',L[3].round(3))

print('Decision Tree          : ',L[4].round(3))

print('Random Forest          : ',L[5].round(3))
# To sum up, 

# Confusion matrixes show there is no unbalanced situation.

# Support Vector Machine and Naive Bayes Classification Algorithms give the best results.