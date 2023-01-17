# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # Plot

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# Importing the dataset
train_dataset = pd.read_csv('../input/train.csv')
test_dataset = pd.read_csv('../input/test.csv')
train_dataset.describe()
train_dataset.head()
test_dataset.head()
y_train = train_dataset.iloc[:, 1].values
#X = dataset[['PassengerId', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Embarked']]
X_train = train_dataset.iloc[:, [0, 2, 4, 5, 6, 7, 11]].values
X_test = test_dataset.iloc[:, [0, 1, 3, 4, 5, 6, 10]].values

m = X_train.shape[0]
family_size_column = np.zeros((m, 1))
X_train = np.append(X_train, family_size_column, axis=1)
X_train[:, 7] = 1 + X_train[:, 4] + X_train[:, 5]
X_train[:, 1] = X_train[:, 1] * X_train[:, 7]
X_train = np.delete(X_train, [4, 5, 7], 1)

m = X_test.shape[0]
family_size_column = np.zeros((m, 1))
X_test = np.append(X_test, family_size_column, axis=1)
X_test[:, 7] = 1 + X_test[:, 4] + X_test[:, 5]
X_test[:, 1] = X_test[:, 1] * X_test[:, 7]
X_test = np.delete(X_test, [4, 5, 7], 1)

m = X_test.shape[0]
pred_column = np.zeros((m, 1))
result = test_dataset.iloc[:, [0]].values
result = np.append(result, pred_column, axis=1)
result = result.astype(int)
result
# NaN value count for Age column in training dataset
nan_age_train = train_dataset[train_dataset['Age'].isnull()]
nan_age_train.shape[0]
# NaN value count for Age column in testing dataset
nan_age_test = test_dataset[test_dataset['Age'].isnull()]
nan_age_test.shape[0]
# NaN value count for Embarked column in training dataset
nan_embarked_train = train_dataset[train_dataset['Embarked'].isnull()]
nan_embarked_train.shape[0]
# NaN value count for Embarked column in testing dataset
nan_embarked_test = test_dataset[test_dataset['Embarked'].isnull()]
nan_embarked_test.shape[0]
# Taking care of missing data
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values = np.nan, strategy= 'mean')

# Fill mean values for Age where age value is NaN
imputer = imputer.fit(X_train[:, 3:4])
X_train[:, 3:4] = imputer.transform(X_train[:, 3:4])

imputer = imputer.fit(X_test[:, 3:4])
X_test[:, 3:4] = imputer.transform(X_test[:, 3:4])

# Fill most_frequent values for Embarked where its value is NaN
#imputer = SimpleImputer(missing_values = np.nan, strategy= 'most_frequent')
#imputer = imputer.fit(X_train[:, 4:5])
#X_train[:, 4:5] = imputer.transform(X_train[:, 4:5])

#imputer = imputer.fit(X_test[:, 4:5])
#X_test[:, 4:5] = imputer.transform(X_test[:, 4:5])
X_train = np.delete(X_train, [4], 1)
X_test = np.delete(X_test, [4], 1)
# Encoding categorical data
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer

# Column transformer for "Sex" and "Embarked" columns
ct = ColumnTransformer(
    [('oh_enc', OneHotEncoder(sparse=False), [2]),],  # the column numbers for which we want to apply this
    remainder='passthrough'  # This leaves the rest of my columns in place
)
X_train = ct.fit_transform(X_train)
X_test = ct.fit_transform(X_test)

labelencoder_y = LabelEncoder()
y_train = labelencoder_y.fit_transform(y_train)
print(X_train[0:5, :])
X_train.shape
print(X_test[0:5, :])
X_test.shape
# Delete dummy variable columns.
X_train = np.delete(X_train, [1], 1)
print(X_train[0:5, :])
X_train.shape
# Delete dummy variable columns.
X_test = np.delete(X_test, [1], 1)
print(X_test[0:5, :])
X_test.shape
# Feature Scalling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
# Fitting the Logistic Regression to the training set.

# Logistic Regression
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
acc_log = round(classifier.score(X_train, y_train) * 100, 2)
acc_log

#result[:, 1] = y_pred
#result
# Support Vector Machines

from sklearn.svm import SVC

svc = SVC()
svc.fit(X_train, y_train)
y_pred = svc.predict(X_test)
acc_svc = round(svc.score(X_train, y_train) * 100, 2)
acc_svc

result[:, 1] = y_pred
result
# k-Nearest Neighbors algorithm (or k-NN for short)
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
acc_knn = round(knn.score(X_train, y_train) * 100, 2)

#result[:, 1] = y_pred
#result
acc_knn
# Linear SVC
from sklearn.svm import LinearSVC

linear_svc = LinearSVC()
linear_svc.fit(X_train, y_train)
y_pred = linear_svc.predict(X_test)
acc_linear_svc = round(linear_svc.score(X_train, y_train) * 100, 2)
acc_linear_svc
# Decision Tree
from sklearn.tree import DecisionTreeClassifier

decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, y_train)
y_pred = decision_tree.predict(X_test)
acc_decision_tree = round(decision_tree.score(X_train, y_train) * 100, 2)
acc_decision_tree
# Random Forest
from sklearn.ensemble import RandomForestClassifier

random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, y_train)
y_pred = random_forest.predict(X_test)
random_forest.score(X_train, y_train)
acc_random_forest = round(random_forest.score(X_train, y_train) * 100, 2)
acc_random_forest

#result[:, 1] = y_pred
#result
# Create a  DataFrame with the passengers ids and our prediction regarding whether they survived or not
submission = pd.DataFrame({'PassengerId':result[:, 0],'Survived':result[:, 1]})

# Visualize the first 5 rows
submission.head()
# Create csv file for submission.
filename = 'Titanic_Survival_Trial_9.csv'
submission.to_csv(filename, index=False)

print('Saved file: ' + filename)