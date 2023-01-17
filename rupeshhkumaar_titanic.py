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
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

train_dataset = pd.read_csv("/kaggle/input/titanic/train.csv")

train_dataset.head()
test_dataset = pd.read_csv("/kaggle/input/titanic/test.csv")

test_dataset.head()
# Making a copy of the dataset for further use

train_data = train_dataset.copy()

test_data = test_dataset.copy()

y = train_data['Survived']

train_data.drop(['Survived'], axis = 1, inplace = True)

X = train_data
# Remove Columns with null values

train_data.isnull().sum()
test_data.isnull().sum()
cols_with_missing = [col for col in X.columns if X[col].isnull().any()]

reduced_X = X.drop(cols_with_missing, axis = 1)

reduced_X.head()
reduced_test = test_data.drop(cols_with_missing, axis = 1)

#Since Fare has one missing value we will fill it by the mean of Fare column

mean = reduced_test["Fare"].mean()

pd.isnull(reduced_test).any(1).nonzero()[0] # To find the array postion of the missing value and put the mean 

reduced_test["Fare"][152] = mean

reduced_test.head()
# Since Name column doesnt have any significance in the rescuing

reduced_test = reduced_test.drop(['Name', 'PassengerId'], axis = 1)

X = X.drop(["Name", "PassengerId"], axis = 1)
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()

encoded_test = pd.DataFrame(encoder.fit_transform(reduced_test["Sex"]))

encoded_X = pd.DataFrame(encoder.fit_transform(X["Sex"]))
# Splitting the Datasets

from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(encoded_X, y, test_size = 0.2, random_state = 0)
# Let's use different Classification algorithms
from sklearn.linear_model import LogisticRegression

classifier_1 = LogisticRegression()

classifier_1.fit(X_train, y_train)

pred_1 = classifier_1.predict(X_val)

classifier_1.fit(encoded_X, y)

predictions = classifier_1.predict(encoded_test)

predictions
# Confusion Matrix

from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score 

from sklearn.metrics import classification_report

cm_1 = confusion_matrix(y_val, pred_1)

print(cm_1)

print("Accuracy", accuracy_score(y_val, pred_1))

print("Classfication Report \n", classification_report(y_val, pred_1))
from sklearn.naive_bayes import GaussianNB

classifier_2 = GaussianNB()

classifier_2.fit(X_train, y_train)

pred_2 = classifier_2.predict(X_val)

classifier_2.fit(encoded_X, y)

predictions = classifier_2.predict(encoded_test)

predictions
from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score 

from sklearn.metrics import classification_report

cm_2 = confusion_matrix(y_val, pred_2)

print(cm_2)

print("Accuracy", accuracy_score(y_val, pred_2))

print("Classfication Report\n", classification_report(y_val, pred_2))
from sklearn.neighbors import KNeighborsClassifier

classifier_3 = KNeighborsClassifier(n_neighbors = 5, metric = "minkowski", p=2)

classifier_3.fit(X_train, y_train)

pred_3 = classifier_3.predict(X_val)
from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score 

from sklearn.metrics import classification_report

cm_3 = confusion_matrix(y_val, pred_3)

print(cm_3)

print("Accuracy", accuracy_score(y_val, pred_3))

print("Classfication Report\n", classification_report(y_val, pred_3))
from sklearn.svm import SVC

classifier_3 = SVC(kernel = "rbf", random_state = 100)

classifier_3.fit(X_train, y_train)

pred_4 = classifier_3.predict(X_val)
from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score 

from sklearn.metrics import classification_report

cm_4 = confusion_matrix(y_val, pred_4)

print(cm_4)

print("Accuracy", accuracy_score(y_val, pred_4))

print("Classfication Report\n", classification_report(y_val, pred_4))
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import mean_absolute_error

classifier_5 = DecisionTreeClassifier(max_leaf_nodes = 1000, random_state = 0, criterion = 'gini')

classifier_5.fit(X_train, y_train)

pred_5 = classifier_5.predict(X_val)
from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score 

from sklearn.metrics import classification_report

cm_5 = confusion_matrix(y_val, pred_5)

print(cm_5)

print("Accuracy", accuracy_score(y_val, pred_5))

print("Classfication Report\n", classification_report(y_val, pred_5))
output = pd.DataFrame({'PassengerId': test_dataset.PassengerId, 'Survived': predictions})

output.to_csv('my_submission.csv', index=False)

print("Your submission was successfully saved!")