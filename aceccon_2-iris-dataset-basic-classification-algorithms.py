# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

%pylab inline



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
#Split data into training and test datasets (training will be based on 70% of data)

from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0) 

#test_size: if integer, number of examples into test dataset; if between 0.0 and 1.0, means proportion

print('There are {} samples in the training set and {} samples in the test set'.format(X_train.shape[0], X_test.shape[0]))
#Scaling data

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import cross_val_score



sc = StandardScaler()

sc.fit(X_train)

X_train_std = sc.transform(X_train)

X_test_std = sc.transform(X_test)



#X_train_std and X_test_std are the scaled datasets to be used in algorithms
#Applying SVC (Support Vector Classification)

from sklearn.svm import SVC



svm = SVC(kernel='rbf', random_state=0, gamma=.10, C=1.0)

svm.fit(X_train_std, y_train)

print('The accuracy of the SVM classifier on training data is {:.2f}'.format(svm.score(X_train_std, y_train)))

print('The accuracy of the SVM classifier on test data is {:.2f}'.format(svm.score(X_test_std, y_test)))
#Applying Knn

from sklearn.neighbors import KNeighborsClassifier



knn = KNeighborsClassifier(n_neighbors = 7, p = 2, metric='minkowski')

knn.fit(X_train_std, y_train)



print('The accuracy of the Knn classifier on training data is {:.2f}'.format(knn.score(X_train_std, y_train)))

print('The accuracy of the Knn classifier on test data is {:.2f}'.format(knn.score(X_test_std, y_test)))
#Applying XGBoost

import xgboost as xgb



xgb_clf = xgb.XGBClassifier()

xgb_clf = xgb_clf.fit(X_train_std, y_train)



print('The accuracy of the XGBoost classifier on training data is {:.2f}'.format(xgb_clf.score(X_train_std, y_train)))

print('The accuracy of the XGBoost classifier on test data is {:.2f}'.format(xgb_clf.score(X_test_std, y_test)))
#Applying Decision Tree

from sklearn import tree



#Create tree object

decision_tree = tree.DecisionTreeClassifier(criterion='gini')



#Train DT based on scaled training set

decision_tree.fit(X_train_std, y_train)



#Print performance

print('The accuracy of the Decision Tree classifier on training data is {:.2f}'.format(decision_tree.score(X_train_std, y_train)))

print('The accuracy of the Decision Tree classifier on test data is {:.2f}'.format(decision_tree.score(X_test_std, y_test)))
#Applying RandomForest

from sklearn.ensemble import RandomForestClassifier



#Create Random Forest object

random_forest = RandomForestClassifier()



#Train model

random_forest.fit(X_train_std, y_train)



#Print performance

print('The accuracy of the Random Forest classifier on training data is {:.2f}'.format(random_forest.score(X_train_std, y_train)))

print('The accuracy of the Random Forest classifier on test data is {:.2f}'.format(random_forest.score(X_test_std, y_test)))
