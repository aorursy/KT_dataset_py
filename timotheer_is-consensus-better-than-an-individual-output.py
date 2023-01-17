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
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

# Encode categorical feature 'Sex'
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
train['Sex'] = labelencoder.fit_transform(train['Sex'])
test['Sex'] = labelencoder.fit_transform(test['Sex'])

# Fill AgeNA with mean
train = train.fillna(29.7)
test = test.fillna(29.7)

# get data for limited list of features only
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']

# get training and test data
X = train[features]
y = train['Survived']

# Get Test and Training data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

# Check if missing values
train.describe()
# Logistic Regression
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression(random_state=0, solver='lbfgs', 
                         multi_class='multinomial').fit(X_train, y_train)
logreg.score(X_test, y_test)
# Naive Bayes Classifier
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB().fit(X_train, y_train)
gnb.score(X_test, y_test)
# Support Vector Machines
from sklearn.svm import SVC
svc = SVC(kernel='linear').fit(X_train, y_train)
svc.score(X_test, y_test)
# Decision Trees
from sklearn import tree
dtc = tree.DecisionTreeClassifier().fit(X_train, y_train)
dtc.score(X_test, y_test)
# Boosted Trees
from sklearn import ensemble
gbc = ensemble.GradientBoostingClassifier().fit(X_train, y_train)
gbc.score(X_test, y_test)
# Random Forest
from sklearn import ensemble
rfc = ensemble.RandomForestClassifier().fit(X_train, y_train)
rfc.score(X_test, y_test)
# Neural Networks
from sklearn import neural_network
mlpc = neural_network.MLPClassifier().fit(X_train, y_train)
mlpc.score(X_test, y_test)
# Nearest Neighbor
from sklearn import neighbors
knn = neighbors.KNeighborsClassifier(n_neighbors=5).fit(X_train, y_train)
knn.score(X_test, y_test)
# What if we merge all the results in one table?
logreg_result = logreg.predict(test[features])
gnb_result = gnb.predict(test[features])
svc_result = svc.predict(test[features])
dtc_result = dtc.predict(test[features])
gbc_result = gbc.predict(test[features])
rfc_result = rfc.predict(test[features])
mlpc_result = mlpc.predict(test[features])
knn_result = knn.predict(test[features])

OutputMatrix = pd.concat([test['PassengerId'],
                          pd.DataFrame(logreg_result),
                          pd.DataFrame(gnb_result),
                          pd.DataFrame(svc_result),
                          pd.DataFrame(dtc_result),
                          pd.DataFrame(gbc_result),
                          pd.DataFrame(rfc_result),
                          pd.DataFrame(mlpc_result),
                          pd.DataFrame(knn_result)],
                         axis=1)

OutputMatrix.columns = ['PassengerId', 'LogReg', 'Gaussian', 'SVC', 'Tree', 'BoostedTree', 'RandomForest', 'NeuralNet', 'KNN']
OutputMatrix['Survived'] = round((OutputMatrix['LogReg']
                                   + OutputMatrix['Gaussian']
                                   + OutputMatrix['SVC']
                                   + OutputMatrix['Tree']
                                   + OutputMatrix['BoostedTree']
                                   + OutputMatrix['RandomForest']
                                   + OutputMatrix['NeuralNet']
                                   + OutputMatrix['KNN']) / 8)

OutputResult = OutputMatrix[['PassengerId', 'Survived']].astype('int64')

# OutputResult.head()
OutputResult.to_csv('Consensus Classifier output.csv', index = False)
# Best is boosted trees
# clf = ensemble.GradientBoostingClassifier().fit(X_train, y_train)
# y_predicted = clf.predict(test[features])

# OutputResult = pd.concat([test['PassengerId'], pd.DataFrame(y_predicted)], axis=1)
# OutputResult.columns = ['PassengerId', 'Survived']
# OutputResult.to_csv('Boosted Trees Classifier output.csv', index = False)
