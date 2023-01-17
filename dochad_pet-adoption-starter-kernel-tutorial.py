# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import matplotlib.pylab as plt
import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train = pd.read_csv("../input/train/train.csv")
test = pd.read_csv("../input/test/test.csv")
color_labels = pd.read_csv("../input/color_labels.csv")
breed_labels = pd.read_csv("../input/breed_labels.csv")
state_labels = pd.read_csv("../input/state_labels.csv")
train.describe()
test.sample(5)
train_simple = pd.read_csv("../input/train/train.csv")
train_simple.drop(['Name', 'RescuerID', 'Description',"PetID"], axis=1, inplace=True)
test.drop(['Name', 'RescuerID', 'Description'], axis=1, inplace=True)
train.drop(['Name', 'RescuerID', 'Description'], axis=1, inplace=True)
train_simple.sample(5)
test.sample(5)
train_simple["NewState"] = train_simple["State"]-41323
train_simple.drop(["State"],inplace = True,axis=1)
test["NewState"] = test["State"]-41323
test.drop(["State"],inplace = True,axis=1)
train_simple.sample(3)
train_simple["Mixed"] = 0
test["Mixed"] = 0
train_simple.sample(5)
indexer = 0 
for x in train_simple["Breed2"]:
    if x > 0:
        train_simple.loc[[indexer],"Mixed"] = 1
        indexer +=1
indexer = 0 
for x in test["Breed2"]:
    if x > 0:
        test.loc[[indexer],"Mixed"] = 1
        indexer +=1
train_simple["Exp"] = 0
for x in range(len(train["PhotoAmt"])):
    train_simple.loc[[x],"Exp"] = train_simple.loc[[x],"PhotoAmt"] + train_simple.loc[[x],"VideoAmt"]
train_simple.drop(["PhotoAmt","VideoAmt"],inplace=True,axis=1)
test["Exp"] = 0
for x in range(len(test["PhotoAmt"])):
    test.loc[[x],"Exp"] = test.loc[[x],"PhotoAmt"] + test.loc[[x],"VideoAmt"]
test.drop(["PhotoAmt","VideoAmt"],inplace=True,axis=1)
indexer = 0 
for x in train_simple["Fee"]:
    if x > 0:
        train_simple.loc[[indexer],"Fee"] = 1
        indexer +=1
    else:
        train_simple.loc[[indexer],"Fee"] = 0
indexer = 0 
for x in test["Fee"]:
    if x > 0:
        test.loc[[indexer],"Fee"] = 1
        indexer += 1
    else:
        test.loc[[indexer],"Fee"] = 0
from sklearn.model_selection import train_test_split

target = train_simple["AdoptionSpeed"]
predict = train_simple.drop(["AdoptionSpeed"],axis = 1)

x_train, x_val, y_train, y_val = train_test_split(predict, target, test_size = 0.22, random_state = 0)
# Gaussian Naive Bayes
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

gaussian = GaussianNB()
gaussian.fit(x_train,y_train)
y_pred = gaussian.predict(x_val)
acc_gaussian = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_gaussian)
# Logistic Regression
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()
logreg.fit(x_train, y_train)
y_pred = logreg.predict(x_val)
acc_logreg = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_logreg)
# Support Vector Machines
from sklearn.svm import SVC

svc = SVC()
svc.fit(x_train, y_train)
y_pred = svc.predict(x_val)
acc_svc = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_svc)
# Linear SVC
from sklearn.svm import LinearSVC

linear_svc = LinearSVC()
linear_svc.fit(x_train, y_train)
y_pred = linear_svc.predict(x_val)
acc_linear_svc = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_linear_svc)
# Perceptron
from sklearn.linear_model import Perceptron

perceptron = Perceptron()
perceptron.fit(x_train, y_train)
y_pred = perceptron.predict(x_val)
acc_perceptron = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_perceptron)
#Decision Tree
from sklearn.tree import DecisionTreeClassifier

decisiontree = DecisionTreeClassifier()
decisiontree.fit(x_train, y_train)
y_pred = decisiontree.predict(x_val)
acc_decisiontree = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_decisiontree)
# Random Forest
from sklearn.ensemble import RandomForestClassifier

randomforest = RandomForestClassifier()
randomforest.fit(x_train, y_train)
y_pred = randomforest.predict(x_val)
acc_randomforest = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_randomforest)
# KNN or k-Nearest Neighbors
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier()
knn.fit(x_train, y_train)
y_pred = knn.predict(x_val)
acc_knn = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_knn)
# Stochastic Gradient Descent
from sklearn.linear_model import SGDClassifier

sgd = SGDClassifier()
sgd.fit(x_train, y_train)
y_pred = sgd.predict(x_val)
acc_sgd = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_sgd)
# Gradient Boosting Classifier
from sklearn.ensemble import GradientBoostingClassifier

gbk = GradientBoostingClassifier()
gbk.fit(x_train, y_train)
y_pred = gbk.predict(x_val)
acc_gbk = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_gbk)
models = pd.DataFrame({
    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 
              'Random Forest', 'Naive Bayes', 'Perceptron', 'Linear SVC', 
              'Decision Tree', 'Stochastic Gradient Descent', 'Gradient Boosting Classifier'],
    'Score': [acc_svc, acc_knn, acc_logreg, 
              acc_randomforest, acc_gaussian, acc_perceptron,acc_linear_svc, acc_decisiontree,
              acc_sgd, acc_gbk]})
models.sort_values(by='Score', ascending=False)
full_train = train_simple.drop(["AdoptionSpeed"],axis = 1)
full_test = train_simple["AdoptionSpeed"]
gbk.fit(full_train, full_test)
sample = pd.read_csv("../input/test/sample_submission.csv")
sample.sample(5)
#set ids as PassengerId and predict survival 
ids = test["PetID"]
predictions = gbk.predict(test.drop(["PetID"],axis = 1))

#set the output as a dataframe and convert to csv file named submission.csv
output = pd.DataFrame({"PetID": ids ,"AdoptionSpeed": predictions})
output.to_csv('submission.csv', index=False)