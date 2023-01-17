import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import os
os.listdir("../input/")
# Show multiple outputs
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
train_dataset = pd.read_csv("../input/train.csv")
test_dataset = pd.read_csv("../input/test.csv")
train_dataset.info()
test_dataset.info()
X_train = train_dataset[["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]].values
y_train = train_dataset[["Survived"]].values
X_test = test_dataset[["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]].values
# Taking care of missing data in train set, Age, Embarked
##  Age imputer by mean
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
X_train[:, [2]] = imputer.fit_transform(X_train[:, [2]])
## Embarked imputer using categorical imputer by most frequent
from sklearn_pandas import CategoricalImputer
imputer = CategoricalImputer()
X_train[:, [-1]] = imputer.fit_transform(X_train[:, [-1]])

# Taking care of missing data in test set, Age, Fare
## Age by mean
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
X_test[:, [2]] = imputer.fit_transform(X_test[:, [2]])
## Fare by mean
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
X_test[:, [-2]] = imputer.fit_transform(X_test[:, [-2]])
from sklearn.preprocessing import LabelEncoder, OneHotEncoder,StandardScaler
# encoding Sex feature
label_enc = LabelEncoder()
X_train[:, 1] = label_enc.fit_transform(X_train[:, 1])
X_test[:, 1] = label_enc.transform(X_test[:, 1])
# encoding Emabarked city feature
label_enc2 = LabelEncoder()
X_train[:, -1] = label_enc2.fit_transform(X_train[:, -1])
X_test[:, -1] = label_enc2.transform(X_test[:, -1])
# one hot encoding Emabarked city feature
oh_enc = OneHotEncoder(categorical_features=[-1])
X_train = oh_enc.fit_transform(X_train).toarray()
X_test = oh_enc.transform(X_test).toarray()

sc_enc = StandardScaler()
X_train = sc_enc.fit_transform(X_train)
X_test = sc_enc.transform(X_test)
# Fitting Logistic Regression to the Training set
from sklearn.svm import SVC
classifier = SVC(kernel = "linear", random_state = 0)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
my_submission = pd.DataFrame({'PassengerId': test_dataset.PassengerId, 'Survived': y_pred})
# you could use any filename. We choose submission here
my_submission.to_csv('submission.csv', index=False)