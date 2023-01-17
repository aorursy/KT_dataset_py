# Logistic Regression



# Importing the libraries

import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

# Importing the dataset

dataset = pd.read_csv('../input/diabetes/diabetes2.csv')
dataset.head()
dataset.isnull().any()
X = dataset.iloc[:, [0,1,2,4,5,6,7]].values

y = dataset.iloc[:, [8]].values
# Splitting the dataset into the Training set and Test set

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 4)
# Feature Scaling

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.transform(X_test)
# Fitting Logistic Regression to the Training set

from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression(random_state = 0)

classifier.fit(X_train, y_train)
# Predicting the Test set results

y_pred = classifier.predict(X_test)

print(y_pred)
# Making the Confusion Matrix

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

print(cm)


from sklearn.metrics import accuracy_score

print("Accuracy is",(accuracy_score(y_test, y_pred)*(100)).round(2))