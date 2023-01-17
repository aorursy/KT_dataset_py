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
train_data = pd.read_csv("/kaggle/input/titanic/train.csv")

test_data = pd.read_csv("/kaggle/input/titanic/test.csv")
train_data.info()
test_data.info()
train_data.describe()
train_data.describe(include=['O'])
combine = [train_data, test_data]
for dataset in combine:

    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
for dataset in combine:

    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col',\

 	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')



    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

    dataset.drop(['PassengerId'], axis=1)

    
pd.crosstab(train_data['Title'], train_data['Sex'])
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}

for dataset in combine:

    dataset['Title'] = dataset['Title'].map(title_mapping)

    dataset['Title'] = dataset['Title'].fillna(0)



train_data.head()
pid = test_data['PassengerId']
train_data = train_data.drop(['Name', 'PassengerId', 'Ticket', 'Cabin'], axis=1)

test_data = test_data.drop(['Name', 'PassengerId', 'Ticket', 'Cabin'], axis=1)

combine = [test_data, train_data]

train_data.shape, test_data.shape
def categorical_imputation(train_data, test_data):

    cols_with_missing = [col for col in train_data.columns

                     if train_data[col].isnull().any()]

    combine = [train_data, test_data]

    for col in cols_with_missing:

        freq = train_data[col].dropna().mode()[0]

        for dataset in combine:

            dataset[f'{col}_was_missing'] = dataset[col].isnull()

            dataset[col] = dataset[col].fillna(freq)



    return train_data, test_data
def numerical_imputation(train_data, test_data):

    return test_data.fillna(test_data.mean()), train_data.fillna(train_data.mean())

    
train_data, test_data = categorical_imputation(train_data, test_data)

#train_data, test_data = numerical_imputation(train_data, test_data)
from sklearn.preprocessing import LabelEncoder

def label_categorical(train_data, test_data):

    s = (train_data.dtypes == 'object')

    object_cols = list(s[s].index)

    print(object_cols)

    labeled_train = train_data.copy()

    labeled_test = test_data.copy()

    label_encoder = LabelEncoder()

    for col in object_cols:

        labeled_train[col] = label_encoder.fit_transform(labeled_train[col])

        labeled_test[col] = label_encoder.transform(labeled_test[col])

    return labeled_train, labeled_test



train_data, test_data = label_categorical(train_data, test_data)
from sklearn.preprocessing import OneHotEncoder

def onehot_categorical(train_data, test_data):

    s = (train_data.dtypes == 'object')

    object_cols = list(s[s].index)

    print(object_cols)

    OH_train = train_data.copy()

    OH_test = test_data.copy()

    OH_encoder = OneHotEncoder()

    for col in object_cols:

        OH_train[col] = pd.DataFrame(OH_encoder.fit_transform(OH_train[col]))

        OH_test[col] = pd.DataFrame(OH_encoder.transform(OH_test[col]))

    return OH_train, OH_test

    

# Apply one-hot encoder to each column with categorical data

#train_data, test_data =  onehot_categorical(train_data, test_data)
from sklearn.model_selection import train_test_split

X = train_data.drop(['Survived'], axis = 1)

y = train_data['Survived']

X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2,

                                                      random_state=0)
from sklearn.metrics import mean_absolute_error

def score_model(model, X_train, X_valid, y_train, y_valid):

    preds = model.predict(X_valid)

    return mean_absolute_error(y_valid, preds)
from sklearn.ensemble import RandomForestClassifier

radom_forest = RandomForestClassifier(n_estimators=100)

radom_forest.fit(X_train, y_train)

radom_forest_score = score_model(radom_forest, X_train, X_valid, y_train, y_valid)

print(radom_forest_score)
from sklearn.svm import SVC, LinearSVC

svc = SVC()

svc.fit(X_train, y_train)

svc_score = score_model(svc, X_train, X_valid, y_train, y_valid)

print(svc_score)
from sklearn.neighbors import KNeighborsClassifier



knn = KNeighborsClassifier(n_neighbors = 7)

knn.fit(X_train, y_train)

knn_score =  score_model(knn, X_train, X_valid, y_train, y_valid)

print(knn_score)
from sklearn.naive_bayes import GaussianNB

gaussian = GaussianNB()

gaussian.fit(X_train, y_train)

gaussian_score =  score_model(gaussian, X_train, X_valid, y_train, y_valid)

print(gaussian_score)
from sklearn.linear_model import Perceptron

perceptron = Perceptron()

perceptron.fit(X_train, y_train)

perceptron_score =  score_model(perceptron, X_train, X_valid, y_train, y_valid)

print(perceptron_score)
linear_svc = LinearSVC(max_iter=120000)

linear_svc.fit(X_train, y_train)

linear_svc_score =  score_model(linear_svc, X_train, X_valid, y_train, y_valid)

print(linear_svc_score)
from sklearn.tree import DecisionTreeClassifier



decision_tree = DecisionTreeClassifier()

decision_tree.fit(X_train, y_train)

decision_tree_score =  score_model(linear_svc, X_train, X_valid, y_train, y_valid)

print(decision_tree_score)
from xgboost import XGBClassifier

xgb_classifier = XGBClassifier()

xgb_classifier.fit(X_train, y_train)

xgb_classifier_score =  score_model(xgb_classifier, X_train, X_valid, y_train, y_valid)

print(xgb_classifier_score)
best_model = XGBClassifier(n_estimators=10000, learning_rate=0.01)

best_model.fit(X, y)
test_data = test_data.fillna(test_data.mean())
predictions = best_model.predict(test_data)



output = pd.DataFrame({'PassengerId': pid, 'Survived': predictions})

output.to_csv('my_submission.csv', index=False)