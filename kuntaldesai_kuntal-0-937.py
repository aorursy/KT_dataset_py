# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import pickle

import os

import sys

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
sys.path.append('/kaggle/input/tools-1/')

sys.path.append('/kaggle/input/tools2/')

 

original = "/kaggle/input/tools-1/final_project_dataset.pkl"

destination = "final_project_dataset_unix.pkl"



content = ''

outsize = 0

with open(original, 'rb') as infile:

    content = infile.read()

with open(destination, 'wb') as output:

    for line in content.splitlines():

        outsize += len(line) + 1

        output.write(line + str.encode('\n'))

# data_dict

data_dict = pickle.load(open(destination, 'rb'))
from feature_format import featureFormat, targetFeatureSplit

from tester import dump_classifier_and_data
my_dataset = data_dict
### Task 1: Select what features you'll use.

### features_list is a list of strings, each of which is a feature name.

### The first feature must be "poi".

features_list = ['poi', 'salary', 'to_messages', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus', 'restricted_stock_deferred', 'deferred_income','total_stock_value', 'expenses','from_poi_to_this_person', 'exercised_stock_options', 'from_messages', 'from_this_person_to_poi', 'long_term_incentive', 'shared_receipt_with_poi', 'restricted_stock', 'director_fees']



### Extract features and labels from dataset for local testing

data = featureFormat(my_dataset, features_list, sort_keys = True)

labels, features = targetFeatureSplit(data)



from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

#features = scaler.fit_transform(features)
## importing all algorithms and creating a classifier

from sklearn.naive_bayes import GaussianNB

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LinearRegression

from sklearn.svm import SVC

from sklearn.ensemble import AdaBoostClassifier

clf_1 = GaussianNB()

clf_2 = DecisionTreeClassifier()

clf_3 = RandomForestClassifier()

clf_4 = SVC(C = 10)

clf_5 = AdaBoostClassifier()
## importing stratifiedKFold and creating skf

from sklearn.model_selection import StratifiedKFold, GridSearchCV

from sklearn.metrics import accuracy_score

from sklearn.metrics import plot_confusion_matrix



skf = StratifiedKFold(n_splits = 9)

features_ = np.array(features)

labels_ = np.array(labels)
## naive_bayes gving a very bad accuracy, so not a good idea using it.

acc_1 = []

#skf_1 = StratifiedKFold(n_splits = 9, shuffle = True)

for train_index, test_index in skf.split(features_, labels_):

    X_train, X_test = features_[train_index], features_[test_index]

    y_train, y_test = labels_[train_index], labels_[test_index]

    clf_1.fit(X_train, y_train)

    score = accuracy_score(y_test, clf_1.predict(X_test))

    acc_1.append(score)



print(acc_1)
## Decision tree classifier has okay accuracy, so will not use.

acc_2 = []

skf_1 = StratifiedKFold(n_splits = 9, shuffle = True)

for train_index, test_index in skf_1.split(features_, labels_):

    X_train, X_test = features_[train_index], features_[test_index]

    y_train, y_test = labels_[train_index], labels_[test_index]

    clf_2.fit(X_train, y_train)

    score = accuracy_score(y_test, clf_2.predict(X_test))

    acc_2.append(score)



print(acc_2)
## Random forest classifier has very good results, but while detecting poi, it is very bad. So will not use.

acc_3 = []

#skf_1 = StratifiedKFold(n_splits = 9, shuffle = True)

for train_index, test_index in skf.split(features_, labels_):

    X_train, X_test = features_[train_index], features_[test_index]

    y_train, y_test = labels_[train_index], labels_[test_index]

    clf_3.fit(X_train, y_train)

    score = accuracy_score(y_test, clf_3.predict(X_test))

    acc_3.append(score)

    #disp = plot_confusion_matrix(clf_3, X_test, y_test)

    #print(disp.confusion_matrix)



print(acc_3)
## SVM has all its accuracy upto 0.875 and is not increasing.

acc_4 = []

skf_1 = StratifiedKFold(n_splits = 9, shuffle = True)

for train_index, test_index in skf_1.split(features_, labels_):

    X_train, X_test = features_[train_index], features_[test_index]

    y_train, y_test = labels_[train_index], labels_[test_index]

    clf_4.fit(X_train, y_train)

    score = accuracy_score(y_test, clf_4.predict(X_test))

    acc_4.append(score)



print(acc_4)
## Adaboost classifer has the best results, to enhance it, we will use this classifier

## We will use feature_importances_(pca tried but didn't worked out).

from sklearn.model_selection import train_test_split

features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.3, random_state=42)



clf_5.fit(features, labels)

print(clf_5.feature_importances_)
import matplotlib.pyplot as plt



for a in range(len(features)) :

    if(labels[a] == True) :

        plt.scatter(features[a][1], features[a][2], color = 'r')

    else : 

        plt.scatter(features[a][1], features[a][2], color = 'b')

plt.xlabel('Total Stock Value')

plt.ylabel('Expenses')

plt.show()
features_list = ['poi', 'salary', 'total_payments', 'bonus', 'deferred_income','total_stock_value', 'expenses', 'exercised_stock_options', 'from_this_person_to_poi', 'shared_receipt_with_poi', 'restricted_stock']

data = featureFormat(my_dataset, features_list, sort_keys = True)

labels, features = targetFeatureSplit(data)



features_ = np.array(features)

labels_ = np.array(labels)



clf_5.fit(features, labels)

print(clf_5.feature_importances_)
acc_5 = []

#skf_1 = StratifiedKFold(n_splits = 9, shuffle = True)

for train_index, test_index in skf.split(features_, labels_):

    X_train, X_test = features_[train_index], features_[test_index]

    y_train, y_test = labels_[train_index], labels_[test_index]

    clf_5.fit(X_train, y_train)

    score = accuracy_score(y_test, clf_5.predict(X_test))

    acc_5.append(score)



print(acc_5)