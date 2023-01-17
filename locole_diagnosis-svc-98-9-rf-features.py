%matplotlib inline

import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn import svm, metrics

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, KFold

from sklearn.preprocessing import LabelEncoder

from sklearn.ensemble import RandomForestClassifier

from scipy.stats import sem
data =  pd.read_csv("../input/data.csv",header = 0)



# list columns

print(data.columns)
# Create Target



y = np.array(data.diagnosis)

labels = LabelEncoder()

target = labels.fit_transform(y)
# Create features normalise the data



cols = data.columns[(data.columns != 'id') & (data.columns != 'diagnosis') & (data.columns != 'Unnamed: 32')]

features = data[cols]

features = (features - features.mean()) / (features.std())
# Look at correlation between features using seaborn



corr = features.corr()

mask = np.zeros_like(corr)

mask[np.triu_indices_from(mask)] = True





with sns.axes_style("white"):

    f, ax = plt.subplots(figsize=(25, 20))

    sns.heatmap(corr, mask=mask, vmax=1, square=True, annot=True)
features = features.drop(labels=['perimeter_mean','area_mean','radius_worst','perimeter_worst', 'area_worst'], axis=1)
# Look at correlation between features using seaborn



corr = features.corr()

mask = np.zeros_like(corr)

mask[np.triu_indices_from(mask)] = True





with sns.axes_style("white"):

    f, ax = plt.subplots(figsize=(25, 20))

    sns.heatmap(corr, mask=mask, vmax=1, square=True, annot=True)
# Split our data into training and test data



X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.33, random_state=123)
# Use grid search to tune hyperparameters for our SVC on training data



svc = svm.SVC()

parameters = [{'kernel':['linear'], 'C': np.logspace(-2, 2, 25)},

             {'kernel':['poly'], 'C': np.logspace(-2,2,25), 'gamma': np.logspace(-4,0,25), 'degree': [2, 3, 4, 5, 6, 7]},

             {'kernel':['rbf', 'sigmoid'], 'C': np.logspace(-2, 2, 25), 'gamma': np.logspace(-4,0,25)}]

clf = GridSearchCV(svc, parameters)

clf.fit(X_train, y_train)
# Print out hyperparameter selections and set-up new classifier with these



print('Best Parameters:', clf.best_params_)

clf = svm.SVC(**clf.best_params_)

clf.fit(X_train, y_train)
# Test our calibrated model accuracy on our test data



print('Accuracy on training data:')

print(clf.score(X_train, y_train))

print('Accuracy on test data:')

print(clf.score(X_test, y_test))



y_pred = clf.predict(X_test)



print('Classification report:')

print(metrics.classification_report(y_test, y_pred))

print('Confusion matrix:')

print(metrics.confusion_matrix(y_test, y_pred))
# Cross validation of support vector classifier



cv = KFold(5, shuffle=True, random_state=123)

scores = cross_val_score(clf, features, target, cv=cv)

print(scores)

print("Mean score: {0:.3f} (+/-{1:.3f})".format(np.mean(scores), sem(scores)))
clf_rf = RandomForestClassifier(n_estimators=2500, max_features=None, criterion='entropy')

clf_rf.fit(X_train, y_train)
# Test our calibrated model accuracy on our test data



print('Accuracy on training data:')

print(clf_rf.score(X_train, y_train))

print('Accuracy on test data:')

print(clf_rf.score(X_test, y_test))



y_pred = clf_rf.predict(X_test)



print('Classification report:')

print(metrics.classification_report(y_test, y_pred))

print('Confusion matrix:')

print(metrics.confusion_matrix(y_test, y_pred))
# Cross validation of random forest



cv = KFold(5, shuffle=True, random_state=123)

scores = cross_val_score(clf_rf, X_train, y_train, cv=cv)

print(scores)

print("Mean score: {0:.3f} (+/-{1:.3f})".format(np.mean(scores), sem(scores)))
importances = clf_rf.feature_importances_

importances = pd.DataFrame(importances, index=features.columns, columns=["Importance"])

importances = importances.sort_values(by=['Importance'],ascending=False)
importances.plot(kind='bar')