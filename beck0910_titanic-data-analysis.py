# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import copy

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
train.head()
train.describe()
missingage = []
for i in range(len(train)):
    if np.isnan(train['Age'][i]):
        missingage.append(train['Parch'][i])
    else:
        pass
sum(missingage)/len(missingage)
train['Age'].mean()
for i in range(len(train)):
    if np.isnan(train['Age'][i]):
        train['Age'][i] = train['Age'].mean()
train['Age'].describe()
train = train.drop(columns = ['Name', 'Ticket', 'PassengerId'])
train['Cabin'].describe()
train = train.drop(columns = ['Cabin'])
train
train['Embarked'].describe()
for i in range(len(train)):
    if (train['Embarked'][i] == 'C'):
        pass
    elif (train['Embarked'][i] == 'S'):
        pass
    elif (train['Embarked'][i] == 'Q'):
        pass
    else:
        train['Embarked'][i] = 'S'
train['Embarked'].describe()
train['EmbarkedC'] = train['Embarked']
train['EmbarkedS'] = train['Embarked']
train['EmbarkedQ'] = train['Embarked']
for i in range(len(train)):
    if (train['Embarked'][i] == 'C'):
        train['EmbarkedC'][i] = 1
    else:
        train['EmbarkedC'][i] = 0
    if (train['Embarked'][i] == 'S'):
        train['EmbarkedS'][i] = 1
    else:
        train['EmbarkedS'][i] = 0
    if (train['Embarked'][i] == 'Q'):
        train['EmbarkedQ'][i] = 1
    else:
        train['EmbarkedQ'][i] = 0
train.head()
train.tail()
train = train.drop(columns = ['Embarked'])
train['SexM'] = train['Sex']
train['SexF'] = train['Sex']
for i in range(len(train)):
    if (train['Sex'][i] == 'male'):
        train['SexM'][i] = 1
    else:
        train['SexM'][i] = 0
    if (train['Sex'][i] == 'female'):
        train['SexF'][i] = 1
    else:
        train['SexF'][i] = 0
train.head()
train = train.drop(columns = ['Sex'])
train['Pclass'].describe()
train['Pclass1'] = train['Pclass']
train['Pclass2'] = train['Pclass']
train['Pclass3'] = train['Pclass']
for i in range(len(train)):
    if (train['Pclass'][i] == 1):
        train['Pclass1'][i] = 1
    else:
        train['Pclass1'][i] = 0
    if (train['Pclass'][i] == 2):
        train['Pclass2'][i] = 1
    else:
        train['Pclass2'][i] = 0
    if (train['Pclass'][i] == 3):
        train['Pclass3'][i] = 1
    else:
        train['Pclass3'][i] = 0
train.head()
train = train.drop(columns = ['Pclass'])
train
training, validation = train_test_split(train, test_size = 0.20)
traininglabels = training['Survived']
validationlabels = validation['Survived']
training = training.drop(columns = ['Survived'])
validation = validation.drop(columns = ['Survived'])
trainPCA = train.copy()
trainPCA2 = trainPCA.copy()
trainPCA2['1'] = trainPCA['Pclass2']
trainPCA2['2'] = trainPCA['Pclass2']
trainPCA2['3'] = trainPCA['Pclass2']
trainPCA2['4'] = trainPCA['Pclass2']
trainPCA2['5'] = trainPCA['Pclass2']
trainPCA2['6'] = trainPCA['Pclass2']
trainPCA2['7'] = trainPCA['Pclass2']
trainPCA2['8'] = trainPCA['Pclass2']
trainPCA2['9'] = trainPCA['Pclass2']
trainPCA2['10'] = trainPCA['Pclass2']
trainPCA2['11'] = trainPCA['Pclass2']
trainPCA2['12'] = trainPCA['Pclass2']
trainPCA2 = trainPCA2.drop(columns = ['Age', 'SibSp', 'Parch', 'Fare', 'EmbarkedC', 'EmbarkedS', 'EmbarkedQ', 'SexM', 'SexF', 'Pclass1', 'Pclass2', 'Pclass3'])
trainPCA = trainPCA.drop(columns = ['Survived'])
pca = PCA()
trainPCA = pca.fit_transform(trainPCA)
trainPCA2['1'] = trainPCA[:,0]
trainPCA2['2'] = trainPCA[:,1]
trainPCA2['3'] = trainPCA[:,2]
trainPCA2['4'] = trainPCA[:,3]
trainPCA2['5'] = trainPCA[:,4]
trainPCA2['6'] = trainPCA[:,5]
trainPCA2['7'] = trainPCA[:,6]
trainPCA2['8'] = trainPCA[:,7]
trainPCA2['9'] = trainPCA[:,8]
trainPCA2['10'] = trainPCA[:,9]
trainPCA2['11'] = trainPCA[:,10]
trainPCA2['12'] = trainPCA[:,11]
trainingPCA, validationPCA = train_test_split(trainPCA2, test_size = 0.20)
trainingPCAlabels = trainingPCA['Survived']
validationPCAlabels = validationPCA['Survived']
trainingPCA = trainingPCA.drop(columns = ['Survived'])
validationPCA = validationPCA.drop(columns = ['Survived'])
print(train)
print(training)
print(trainingPCA)
trainSTD = train.copy()
trainSTD2 = trainSTD.copy()
trainSTD = trainSTD.drop(columns = ['Survived'])
scaler = StandardScaler()
trainSTD = scaler.fit_transform(trainSTD)
trainSTD2['Age'] = trainSTD[:,0]
trainSTD2['SibSp'] = trainSTD[:,1]
trainSTD2['Parch'] = trainSTD[:,2]
trainSTD2['Fare'] = trainSTD[:,3]
trainSTD2['EmbarkedC'] = trainSTD[:,4]
trainSTD2['EmbarkedS'] = trainSTD[:,5]
trainSTD2['EmbarkedQ'] = trainSTD[:,6]
trainSTD2['SexM'] = trainSTD[:,7]
trainSTD2['SexF'] = trainSTD[:,8]
trainSTD2['Pclass1'] = trainSTD[:,9]
trainSTD2['Pclass2'] = trainSTD[:,10]
trainSTD2['Pclass3'] = trainSTD[:,11]
trainingSTD, validationSTD = train_test_split(trainSTD2, test_size = 0.20)
trainingSTDlabels = trainingSTD['Survived']
validationSTDlabels = validationSTD['Survived']
trainingSTD = trainingSTD.drop(columns = ['Survived'])
validationSTD = validationSTD.drop(columns = ['Survived'])
print(train)
print(training)
print(trainingPCA)
print(trainingSTD)
rfclf = RandomForestClassifier(n_estimators = 100)
rfclf.fit(training, traininglabels)
featureimportances = rfclf.feature_importances_
featureimportances
training.head()
rfclf.score(validation, validationlabels)
rfclf2 = RandomForestClassifier(n_estimators = 100)
rfclf2.fit(trainingPCA, trainingPCAlabels)
rfclf2.score(validationPCA, validationPCAlabels)
rfclf3 = RandomForestClassifier(n_estimators = 100)
rfclf3.fit(trainingSTD, trainingSTDlabels)

rfclf3.score(validationSTD, validationSTDlabels)
mlpclf = MLPClassifier(max_iter = 1000)
mlpclf.fit(training, traininglabels)
mlpclf.score(validation, validationlabels)
mlpclf2 = MLPClassifier(max_iter = 1000)
mlpclf2.fit(trainingPCA, trainingPCAlabels)
mlpclf2.score(validationPCA, validationPCAlabels)
mlpclf3 = MLPClassifier(max_iter = 10000)
mlpclf3.fit(trainingSTD, trainingSTDlabels)

mlpclf3.score(validationSTD, validationSTDlabels)
lrclf = LogisticRegression(solver = 'lbfgs', max_iter = 1000)
lrclf.fit(training, traininglabels)
lrclf.score(validation, validationlabels)
lrclf2 = LogisticRegression(solver = 'lbfgs', max_iter = 1000)
lrclf2.fit(trainingPCA, trainingPCAlabels)
lrclf2.score(validationPCA, validationPCAlabels)
lrclf3 = LogisticRegression(solver = 'lbfgs', max_iter = 1000)
lrclf3.fit(trainingSTD, trainingSTDlabels)

lrclf3.score(validationSTD, validationSTDlabels)
svclf = SVC(gamma = 'auto')
svclf.fit(training, traininglabels)
svclf.score(validation, validationlabels)
svclf2 = SVC(gamma = 'auto')
svclf2.fit(trainingPCA, trainingPCAlabels)
svclf2.score(validationPCA, validationPCAlabels)
svclf3 = SVC(gamma = 'auto')
svclf3.fit(trainingSTD, trainingSTDlabels)

svclf3.score(validationSTD, validationSTDlabels)
knnclf = KNeighborsClassifier(n_neighbors = 5)
knnclf.fit(training, traininglabels)
knnclf.score(validation, validationlabels)
knnclf2 = KNeighborsClassifier(n_neighbors = 5)
knnclf2.fit(trainingPCA, trainingPCAlabels)
knnclf2.score(validationPCA, validationPCAlabels)
knnclf3 = KNeighborsClassifier(n_neighbors = 5)
knnclf3.fit(trainingSTD, trainingSTDlabels)

knnclf3.score(validationSTD, validationSTDlabels)
gnbclf = GaussianNB()
gnbclf.fit(training, traininglabels)
gnbclf.score(validation, validationlabels)
gnbclf2 = GaussianNB()
gnbclf2.fit(trainingPCA, trainingPCAlabels)
gnbclf2.score(validationPCA, validationPCAlabels)
gnbclf3 = GaussianNB()
gnbclf3.fit(trainingSTD, trainingSTDlabels)

gnbclf3.score(validationSTD, validationSTDlabels)
sgdclf = SGDClassifier(max_iter = 1000)
sgdclf.fit(training, traininglabels)
sgdclf.score(validation, validationlabels)
sgdclf2 = SGDClassifier(max_iter = 1000)
sgdclf2.fit(trainingPCA, trainingPCAlabels)
sgdclf2.score(validationPCA, validationPCAlabels)
sgdclf3 = SGDClassifier(max_iter = 1000)
sgdclf3.fit(trainingSTD, trainingSTDlabels)

sgdclf3.score(validationSTD, validationSTDlabels)
training, validation = train_test_split(train, test_size = 0.20)
traininglabels = training['Survived']
validationlabels = validation['Survived']
training = training.drop(columns = ['Survived'])
validation = validation.drop(columns = ['Survived'])
test = [10, 50, 100, 200, 500]
for i in test:
    rfclf = RandomForestClassifier(n_estimators = i)
    rfclf.fit(training, traininglabels)
    print("For n_estimators = " + str(i) + ", the accuracy is: " + str(rfclf.score(validation, validationlabels)))
rfclf = RandomForestClassifier(n_estimators = 100, criterion = "entropy")
rfclf.fit(training, traininglabels)
rfclf.score(validation, validationlabels)
rfclf = RandomForestClassifier(n_estimators = 100, criterion = "entropy", class_weight = "balanced")
rfclf.fit(training, traininglabels)
rfclf.score(validation, validationlabels)
training, validation = train_test_split(train, test_size = 0.20)
traininglabels = training['Survived']
validationlabels = validation['Survived']
training = training.drop(columns = ['Survived'])
validation = validation.drop(columns = ['Survived'])
test = [100]
for i in test:
    mlpclf = MLPClassifier(hidden_layer_sizes = (i,), max_iter = 1000)
    mlpclf.fit(training, traininglabels)
    print("For hidden_layer_sizes = " + str(i) + ", the accuracy is: " + str(mlpclf.score(validation, validationlabels)))
i = "lbfgs"
mlpclf = MLPClassifier(solver = i, max_iter = 1000)
mlpclf.fit(training, traininglabels)
print("For solver = " + str(i) + ", the accuracy is: " + str(mlpclf.score(validation, validationlabels)))
i = "sgd"
mlpclf = MLPClassifier(solver = i, max_iter = 1000)
mlpclf.fit(training, traininglabels)
print("For solver = " + str(i) + ", the accuracy is: " + str(mlpclf.score(validation, validationlabels)))
i = "adam"
mlpclf = MLPClassifier(solver = i, max_iter = 1000)
mlpclf.fit(training, traininglabels)
print("For solver = " + str(i) + ", the accuracy is: " + str(mlpclf.score(validation, validationlabels)))
trainSTD = train.copy()
trainSTD2 = trainSTD.copy()
trainSTD = trainSTD.drop(columns = ['Survived'])
scaler = StandardScaler()
trainSTD = scaler.fit_transform(trainSTD)
trainSTD2['Age'] = trainSTD[:,0]
trainSTD2['SibSp'] = trainSTD[:,1]
trainSTD2['Parch'] = trainSTD[:,2]
trainSTD2['Fare'] = trainSTD[:,3]
trainSTD2['EmbarkedC'] = trainSTD[:,4]
trainSTD2['EmbarkedS'] = trainSTD[:,5]
trainSTD2['EmbarkedQ'] = trainSTD[:,6]
trainSTD2['SexM'] = trainSTD[:,7]
trainSTD2['SexF'] = trainSTD[:,8]
trainSTD2['Pclass1'] = trainSTD[:,9]
trainSTD2['Pclass2'] = trainSTD[:,10]
trainSTD2['Pclass3'] = trainSTD[:,11]
trainingSTD, validationSTD = train_test_split(trainSTD2, test_size = 0.20)
trainingSTDlabels = trainingSTD['Survived']
validationSTDlabels = validationSTD['Survived']
trainingSTD = trainingSTD.drop(columns = ['Survived'])
validationSTD = validationSTD.drop(columns = ['Survived'])
i = "linear"
svclf = SVC(gamma = "auto", kernel = i)
svclf.fit(trainingSTD, trainingSTDlabels)
print("For kernel = " + str(i) + ", the accuracy is: " + str(svclf.score(validationSTD, validationSTDlabels)))
i = "poly"
svclf = SVC(gamma = "auto", kernel = i)
svclf.fit(trainingSTD, trainingSTDlabels)
print("For kernel = " + str(i) + ", the accuracy is: " + str(svclf.score(validationSTD, validationSTDlabels)))
i = "rbf"
svclf = SVC(gamma = "auto", kernel = i)
svclf.fit(trainingSTD, trainingSTDlabels)
print("For kernel = " + str(i) + ", the accuracy is: " + str(svclf.score(validationSTD, validationSTDlabels)))
i = "sigmoid"
svclf = SVC(gamma = "auto", kernel = i)
svclf.fit(trainingSTD, trainingSTDlabels)
print("For kernel = " + str(i) + ", the accuracy is: " + str(svclf.score(validationSTD, validationSTDlabels)))
test = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
for i in test:
    svclf = SVC(gamma = "auto", kernel = "poly", degree = i)
    svclf.fit(trainingSTD, trainingSTDlabels)
    print("For degree = " + str(i) + ", the accuracy is: " + str(svclf.score(validationSTD, validationSTDlabels)))
svclf = SVC(gamma = "scale", kernel = "poly")
svclf.fit(trainingSTD, trainingSTDlabels)
print("For gamme = scale, the accuracy is: " + str(svclf.score(validationSTD, validationSTDlabels)))
svclf = SVC(gamma = "auto", kernel = "poly")
svclf.fit(trainingSTD, trainingSTDlabels)
print("For gamme = auto, the accuracy is: " + str(svclf.score(validationSTD, validationSTDlabels)))
svclf = SVC(gamma = "scale", kernel = "poly")
svclf.fit(trainingSTD, trainingSTDlabels)
print("For class_weight = auto, the accuracy is: " + str(svclf.score(validationSTD, validationSTDlabels)))
svclf = SVC(gamma = "scale", kernel = "poly", class_weight = "balanced")
svclf.fit(trainingSTD, trainingSTDlabels)
print("For class_weight = balanced, the accuracy is: " + str(svclf.score(validationSTD, validationSTDlabels)))
test = pd.read_csv("../input/test.csv")
test
for i in range(len(test)):
    if np.isnan(test['Age'][i]):
        test['Age'][i] = train['Age'].mean()
for i in range(len(test)):
    if np.isnan(test['Fare'][i]):
        test['Fare'][i] = train['Fare'].mean()
test = test.drop(columns = ['Name', 'Ticket', 'Cabin'])
for i in range(len(test)):
    if (test['Embarked'][i] == 'C'):
        pass
    elif (test['Embarked'][i] == 'S'):
        pass
    elif (test['Embarked'][i] == 'Q'):
        pass
    else:
        test['Embarked'][i] = 'S'
test['EmbarkedC'] = test['Embarked']
test['EmbarkedS'] = test['Embarked']
test['EmbarkedQ'] = test['Embarked']
for i in range(len(test)):
    if (test['Embarked'][i] == 'C'):
        test['EmbarkedC'][i] = 1
    else:
        test['EmbarkedC'][i] = 0
    if (test['Embarked'][i] == 'S'):
        test['EmbarkedS'][i] = 1
    else:
        test['EmbarkedS'][i] = 0
    if (test['Embarked'][i] == 'Q'):
        test['EmbarkedQ'][i] = 1
    else:
        test['EmbarkedQ'][i] = 0
test = test.drop(columns = ['Embarked'])
test['SexM'] = test['Sex']
test['SexF'] = test['Sex']
for i in range(len(test)):
    if (test['Sex'][i] == 'male'):
        test['SexM'][i] = 1
    else:
        test['SexM'][i] = 0
    if (test['Sex'][i] == 'female'):
        test['SexF'][i] = 1
    else:
        test['SexF'][i] = 0
test = test.drop(columns = ['Sex'])
test['Pclass1'] = test['Pclass']
test['Pclass2'] = test['Pclass']
test['Pclass3'] = test['Pclass']
for i in range(len(test)):
    if (test['Pclass'][i] == 1):
        test['Pclass1'][i] = 1
    else:
        test['Pclass1'][i] = 0
    if (test['Pclass'][i] == 2):
        test['Pclass2'][i] = 1
    else:
        test['Pclass2'][i] = 0
    if (test['Pclass'][i] == 3):
        test['Pclass3'][i] = 1
    else:
        test['Pclass3'][i] = 0
test = test.drop(columns = ['Pclass'])
test
ID = test["PassengerId"]
test = test.drop(columns = ["PassengerId"])
rfclf = RandomForestClassifier(n_estimators = 100, criterion = "entropy", class_weight = "balanced")
rfclf.fit(training, traininglabels)
rfclf.score(validation, validationlabels)
test['Survived'] = rfclf.predict(test)
test['PassengerId'] = ID
Survived = test['Survived']
test = test.drop(columns = ['Survived'])
test
test['Survived'] = Survived
test.head()
test = test.drop(columns = ['Age', 'SibSp', 'Parch', 'Fare', 'EmbarkedC', 'EmbarkedS', 'EmbarkedC', 'EmbarkedQ', 'SexM', 'SexF', 'Pclass1', 'Pclass2', 'Pclass3'])
test.head()
submission = test
submission.to_csv('submission1.csv', index = False)
test = pd.read_csv("../input/test.csv")
for i in range(len(test)):
    if np.isnan(test['Age'][i]):
        test['Age'][i] = train['Age'].mean()
for i in range(len(test)):
    if np.isnan(test['Fare'][i]):
        test['Fare'][i] = train['Fare'].mean()
test = test.drop(columns = ['Name', 'Ticket', 'Cabin'])
for i in range(len(test)):
    if (test['Embarked'][i] == 'C'):
        pass
    elif (test['Embarked'][i] == 'S'):
        pass
    elif (test['Embarked'][i] == 'Q'):
        pass
    else:
        test['Embarked'][i] = 'S'
test['EmbarkedC'] = test['Embarked']
test['EmbarkedS'] = test['Embarked']
test['EmbarkedQ'] = test['Embarked']
for i in range(len(test)):
    if (test['Embarked'][i] == 'C'):
        test['EmbarkedC'][i] = 1
    else:
        test['EmbarkedC'][i] = 0
    if (test['Embarked'][i] == 'S'):
        test['EmbarkedS'][i] = 1
    else:
        test['EmbarkedS'][i] = 0
    if (test['Embarked'][i] == 'Q'):
        test['EmbarkedQ'][i] = 1
    else:
        test['EmbarkedQ'][i] = 0
test = test.drop(columns = ['Embarked'])
test['SexM'] = test['Sex']
test['SexF'] = test['Sex']
for i in range(len(test)):
    if (test['Sex'][i] == 'male'):
        test['SexM'][i] = 1
    else:
        test['SexM'][i] = 0
    if (test['Sex'][i] == 'female'):
        test['SexF'][i] = 1
    else:
        test['SexF'][i] = 0
test = test.drop(columns = ['Sex'])
test['Pclass1'] = test['Pclass']
test['Pclass2'] = test['Pclass']
test['Pclass3'] = test['Pclass']
for i in range(len(test)):
    if (test['Pclass'][i] == 1):
        test['Pclass1'][i] = 1
    else:
        test['Pclass1'][i] = 0
    if (test['Pclass'][i] == 2):
        test['Pclass2'][i] = 1
    else:
        test['Pclass2'][i] = 0
    if (test['Pclass'][i] == 3):
        test['Pclass3'][i] = 1
    else:
        test['Pclass3'][i] = 0
test = test.drop(columns = ['Pclass'])
ID = test["PassengerId"]
test = test.drop(columns = ["PassengerId"])
mlpclf = MLPClassifier(max_iter = 1000)
mlpclf.fit(training, traininglabels)
test['Survived'] = mlpclf.predict(test)
test['PassengerId'] = ID
Survived = test['Survived']
test = test.drop(columns = ['Survived'])
test['Survived'] = Survived
test = test.drop(columns = ['Age', 'SibSp', 'Parch', 'Fare', 'EmbarkedC', 'EmbarkedS', 'EmbarkedC', 'EmbarkedQ', 'SexM', 'SexF', 'Pclass1', 'Pclass2', 'Pclass3'])
submission2 = test
submission2.to_csv('submission2.csv', index = False)
test = pd.read_csv("../input/test.csv")
for i in range(len(test)):
    if np.isnan(test['Age'][i]):
        test['Age'][i] = train['Age'].mean()
for i in range(len(test)):
    if np.isnan(test['Fare'][i]):
        test['Fare'][i] = train['Fare'].mean()
test = test.drop(columns = ['Name', 'Ticket', 'Cabin'])
for i in range(len(test)):
    if (test['Embarked'][i] == 'C'):
        pass
    elif (test['Embarked'][i] == 'S'):
        pass
    elif (test['Embarked'][i] == 'Q'):
        pass
    else:
        test['Embarked'][i] = 'S'
test['EmbarkedC'] = test['Embarked']
test['EmbarkedS'] = test['Embarked']
test['EmbarkedQ'] = test['Embarked']
for i in range(len(test)):
    if (test['Embarked'][i] == 'C'):
        test['EmbarkedC'][i] = 1
    else:
        test['EmbarkedC'][i] = 0
    if (test['Embarked'][i] == 'S'):
        test['EmbarkedS'][i] = 1
    else:
        test['EmbarkedS'][i] = 0
    if (test['Embarked'][i] == 'Q'):
        test['EmbarkedQ'][i] = 1
    else:
        test['EmbarkedQ'][i] = 0
test = test.drop(columns = ['Embarked'])
test['SexM'] = test['Sex']
test['SexF'] = test['Sex']
for i in range(len(test)):
    if (test['Sex'][i] == 'male'):
        test['SexM'][i] = 1
    else:
        test['SexM'][i] = 0
    if (test['Sex'][i] == 'female'):
        test['SexF'][i] = 1
    else:
        test['SexF'][i] = 0
test = test.drop(columns = ['Sex'])
test['Pclass1'] = test['Pclass']
test['Pclass2'] = test['Pclass']
test['Pclass3'] = test['Pclass']
for i in range(len(test)):
    if (test['Pclass'][i] == 1):
        test['Pclass1'][i] = 1
    else:
        test['Pclass1'][i] = 0
    if (test['Pclass'][i] == 2):
        test['Pclass2'][i] = 1
    else:
        test['Pclass2'][i] = 0
    if (test['Pclass'][i] == 3):
        test['Pclass3'][i] = 1
    else:
        test['Pclass3'][i] = 0
test = test.drop(columns = ['Pclass'])
ID = test["PassengerId"]
test = test.drop(columns = ["PassengerId"])
test = scaler.transform(test)
svclf = SVC(gamma = "scale", kernel = "poly")
svclf.fit(trainingSTD, trainingSTDlabels)
predictions = svclf.predict(test)
predictions = predictions.tolist()
type(predictions)
submission3 = pd.DataFrame()
submission3['PassengerId'] = ID
submission3['Survived'] = predictions
submission3
submission3.to_csv('submission3.csv', index = False)
