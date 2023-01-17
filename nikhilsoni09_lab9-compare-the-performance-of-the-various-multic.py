import pandas as pd

import numpy as np  

import seaborn as sns 

from sklearn.model_selection import train_test_split 

from sklearn.ensemble import RandomForestClassifier

import sklearn.metrics as metrics

from sklearn.svm import SVC
df = pd.read_csv("../input/letterrecognition/letter-recognition.data",header = None)
df.head()
df.shape
df.describe()
df.columns
df.info()
sns.countplot(df.iloc[:,0])
X_train, X_test, y_train, y_test = train_test_split(df.iloc[:,1:], df.iloc[:,0], test_size = 0.3)
X_train.shape,X_test.shape,y_train.shape,y_test.shape
# non-linear model

# using rbf kernel, C=1, default value of gamma



# model

non_linear_model = SVC(kernel='rbf')



# fit

non_linear_model.fit(X_train, y_train)



# predict

y_pred = non_linear_model.predict(X_test)
# confusion matrix and accuracy



# accuracy

print("Random Forest Metrics:","\n")

print("accuracy:", metrics.accuracy_score(y_true=y_test, y_pred=y_pred), "\n")

print("micro recall:", metrics.recall_score(y_true=y_test, y_pred=y_pred, average = 'micro'), "\n")

print("macro recall:", metrics.recall_score(y_true=y_test, y_pred=y_pred, average = 'macro'), "\n")

print("micro precision:", metrics.precision_score(y_true=y_test, y_pred=y_pred, average = 'micro'), "\n")

print("macro precision:", metrics.precision_score(y_true=y_test, y_pred=y_pred, average = 'macro'), "\n")

# cm

print(metrics.confusion_matrix(y_true=y_test, y_pred=y_pred))
classifier = RandomForestClassifier(n_estimators = 100, criterion = 'entropy')

classifier.fit(X_train, y_train)



y_pred = classifier.predict(X_test)
# confusion matrix and accuracy



# accuracy

print("Random Forest Metrics:","\n")

print("accuracy:", metrics.accuracy_score(y_true=y_test, y_pred=y_pred), "\n")

print("micro recall:", metrics.recall_score(y_true=y_test, y_pred=y_pred, average = 'micro'), "\n")

print("macro recall:", metrics.recall_score(y_true=y_test, y_pred=y_pred, average = 'macro'), "\n")

print("micro precision:", metrics.precision_score(y_true=y_test, y_pred=y_pred, average = 'micro'), "\n")

print("macro precision:", metrics.precision_score(y_true=y_test, y_pred=y_pred, average = 'macro'), "\n")

# cm

print(metrics.confusion_matrix(y_true=y_test, y_pred=y_pred))