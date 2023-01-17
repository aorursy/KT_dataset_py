# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")
train.head()
test.head()
train.describe()
test.describe()
import seaborn as sns

import matplotlib.pyplot as plt



from sklearn.preprocessing import LabelEncoder

from sklearn.cross_validation import StratifiedShuffleSplit
def encode(train,test):

    le = LabelEncoder().fit(train.species)

    labels = le.transform(train.species)

    classes = list(le.classes_)

    test_ids = test.id

    

    train = train.drop(['species','id'], axis = 1)

    

    return train, labels, test, test_ids, classes



train,labels, test, test_ids, classes = encode(train,test)

train.head(1)
sss = StratifiedShuffleSplit(labels,10, test_size=0.2, random_state=23)

for train_index, test_index in sss:

    X_train, X_test = train.values[train_index], train.values[test_index]

    y_train, y_test = labels[train_index], labels[test_index]
from sklearn.metrics import accuracy_score, log_loss

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC, LinearSVC, NuSVC

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier,

GradientBoostingClassifier)



from sklearn.naive_bayes import GaussianNB

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
classifiers = [KNeighborsClassifier(3),

              SVC(kernel='rbf', C=0.025, probability=True),

              NuSVC(probability=True),

              DecisionTreeClassifier(),

              RandomForestClassifier(),

              AdaBoostClassifier(),

              GradientBoostingClassifier(),

              GaussianNB(),

              LinearDiscriminantAnalysis(),

              QuadraticDiscriminantAnalysis()]



log_cols = ["Classifier","Accuracy","Log Loss"]

log = pd.DataFrame(columns=log_cols)



for clf in classifiers:

    clf.fit(X_train,y_train)

    name = clf.__class__.__name__

    

    print("="*30)

    print(name)

    

    print("***Results***")

    

    train_predictions = clf.predict(X_test)

    acc = accuracy_score(y_test, train_predictions)

    print("Accuracy: {:.4%}".format(acc))

    

    train_predictions = clf.predict_proba(X_test)

    ll = log_loss(y_test, train_predictions)

    print("Log Loss: {}".format(ll))

    

    log_entry = pd.DataFrame([[name,acc*100, ll]], columns=log_cols)

    log = log.append(log_entry)

    print("="*30)

    