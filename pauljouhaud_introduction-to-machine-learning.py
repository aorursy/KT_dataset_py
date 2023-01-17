# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from matplotlib import pyplot as plt

import seaborn as sns

from sklearn import tree

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier, RandomForestClassifier

from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from time import time

import csv



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.



# Display Matplotlib diagrams inside the notebook

%matplotlib inline
data = pd.read_csv('../input/train.csv')
data.Survived = data.Survived.map({False: 'Deceased', True: 'Survived'})
data.Age = data.Age.fillna(data.Age.mean())
dummy_sex = pd.Series(np.where(data.Sex == 'female', 1, 0) , name='Sex')

dummy_embarked = pd.get_dummies(data.Embarked , prefix='Embarked')

dummy_class = pd.get_dummies(data.Pclass, prefix='Class')
cleaned_data = pd.concat([dummy_sex, dummy_embarked, dummy_class, data.Age, data.Fare, data.SibSp, data.Parch], axis=1)
def clean_dataset(dataset):

    dataset.Age = dataset.Age.fillna(dataset.Age.mean())

    dataset.Fare = dataset.Fare.fillna(dataset.Fare.mean())

    dummy_sex = pd.Series(np.where(dataset.Sex == 'female', 1, 0) , name='Sex')

    dummy_embarked = pd.get_dummies(dataset.Embarked , prefix='Embarked')

    dummy_class = pd.get_dummies(dataset.Pclass, prefix='Class')

    cleaned_data = pd.concat([dummy_sex, dummy_embarked, dummy_class, dataset.Age, dataset.Fare, dataset.SibSp, dataset.Parch], axis=1)

    return cleaned_data
classifiers = [

    tree.DecisionTreeClassifier(), 

    tree.DecisionTreeClassifier(max_leaf_nodes=12), 

    BaggingClassifier(n_estimators=50),

    RandomForestClassifier(),

    tree.ExtraTreeClassifier(),

    AdaBoostClassifier(n_estimators=50, algorithm='SAMME'),

    KNeighborsClassifier(n_neighbors = 3),

    GaussianNB(),

    LogisticRegression(),

    LinearSVC(),

    SVC()

]
X_train, X_test, y_train, y_test = train_test_split(

    cleaned_data, data.Survived, test_size=0.25, random_state=42)
def compute_empirical_risk(y_test, y_pred):

    sum = 0

    test = y_test.values

    for i in list(range(0, len(test))):

        if(test[i] != y_pred[i]):

            sum = sum + 1

    return sum/len(y_test) * 100
t0_total = time()

best_clf = None

best_risk = 100

best_accuracy = 0

ensemble_pred = []

length_pred = len(y_test)

for clf in classifiers:

    print(clf)

    t0 = time()

    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    ensemble_pred.append(y_pred)

    t1 = time()

    print('Done in %0.3f seconds' % (t1-t0))

    empirical_risk = compute_empirical_risk(y_test=y_test, y_pred=y_pred)

    accuracy =  accuracy_score(y_test, y_pred) * 100

    print("Empirical risk: {0:.2f}%".format(empirical_risk))

    print("Accuracy score: {0:.2f}%".format(accuracy))

    print(classification_report(y_test, y_pred, target_names=['Deceased', 'Survived']))

    if empirical_risk < best_risk:

        best_risk = empirical_risk

        best_clf = clf

        best_accuracy = accuracy

    print('====================================================\n\n')

t1_total = time()

print('All done in %0.3f seconds' % (t1_total-t0_total))



print('The Best Classifier is:')

print(best_clf)

print('With an empirical risk of {0:.2f}%'.format(best_risk))

print('And an accuracy score of {0:.2f}%'.format(best_accuracy))
y_pred = []

for i in range(length_pred):

    pred = 0

    for y in ensemble_pred:

        if y[i] == 'Survived':

            pred = pred + 1

    pred = pred / len(ensemble_pred)

    if pred > 0.5:

        y_pred.append('Survived')

    else:

        y_pred.append('Deceased')
print(classification_report(y_test, y_pred, target_names=['Deceased', 'Survived']))

empirical_risk = compute_empirical_risk(y_test=y_test, y_pred=y_pred)

accuracy = accuracy_score(y_test, y_pred) * 100

print("Empirical risk: {0:.2f}%".format(empirical_risk))

print("Accuracy score: {0:.2f}%".format(accuracy))
test = pd.read_csv('../input/test.csv')
test.info()
cleaned_test = clean_dataset(test)

y_pred = best_clf.predict(cleaned_test)
for i in range(len(y_pred)):

    if y_pred[i] == 'Survived':

        y_pred[i] = True

    else:

        y_pred[i] = False
submission = pd.DataFrame({'PassengerId': test.PassengerId, 'Survived': y_pred})

submission.to_csv("submission.csv", index=False)