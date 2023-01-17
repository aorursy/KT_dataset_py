# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

%matplotlib inline

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train_df = pd.read_csv("../input/titanicprocesseddata/data.csv") 
train_df.head()
train_df.drop(['Unnamed: 0'], axis=1,inplace=True)
train_df.head()
test_df = pd.read_csv("../input/titanic/test.csv") 
test_df.head()
# Check for null values in columns
test_df.isnull().sum()
test_df['Age'].fillna((train_df['Age'].median()), inplace=True)
test_df['Fare'].fillna((train_df['Fare'].median()), inplace=True)
test_df['Cabin_Available'] = np.where(test_df.Cabin.notna(), 1, 0)

test_df.isnull().sum()
feature_names = ['Age', 'Gender_male', 'Gender_female', 'Pclass','SibSp','Parch','Fare','Cabin_Available', 'EmbarkedFrom_C','EmbarkedFrom_Q','EmbarkedFrom_S']
X = train_df[feature_names]
y = train_df['Survived']

# one-hot encoding test categorical data retaining the Sex & Embarked columns 
test_df['Gender'] = test_df['Sex']
test_df['EmbarkedFrom'] = test_df['Embarked']
test_df = pd.get_dummies(test_df, columns=['Gender','EmbarkedFrom'])
test_df.head()

X_test_final = test_df[feature_names]
X_test_final[0:1]
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
X_test_final = scaler.transform(X_test_final)
X_test_final[:1]
# Logistic Regression
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
print('Accuracy of Logistic regression classifier on training set: {:.2f}'
     .format(logreg.score(X_train, y_train)))
print('Accuracy of Logistic regression classifier on test set: {:.2f}'
     .format(logreg.score(X_test, y_test)))
# Decision Tree
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier().fit(X_train, y_train)
print('Accuracy of Decision Tree classifier on training set: {:.2f}'
     .format(clf.score(X_train, y_train)))
print('Accuracy of Decision Tree classifier on test set: {:.2f}'
     .format(clf.score(X_test, y_test)))
# K-Nearest Neighbour (KNN)
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
print('Accuracy of K-NN classifier on training set: {:.2f}'
     .format(knn.score(X_train, y_train)))
print('Accuracy of K-NN classifier on test set: {:.2f}'
     .format(knn.score(X_test, y_test)))
# Gaussian Naive Bayes
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train, y_train)
print('Accuracy of GNB classifier on training set: {:.2f}'
     .format(gnb.score(X_train, y_train)))
print('Accuracy of GNB classifier on test set: {:.2f}'
     .format(gnb.score(X_test, y_test)))
# SVM
from sklearn.svm import SVC
svm = SVC()
svm.fit(X_train, y_train)
print('Accuracy of SVM classifier on training set: {:.2f}'
     .format(svm.score(X_train, y_train)))
print('Accuracy of SVM classifier on test set: {:.2f}'
     .format(svm.score(X_test, y_test)))
# Random forest.
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_jobs=10, random_state=0)

rf.fit(X_train, y_train)
print('Accuracy of Random Forest classifier on training set: {:.2f}'
     .format(rf.score(X_train, y_train)))
print('Accuracy of Random Forest classifier on test set: {:.2f}'
     .format(rf.score(X_test, y_test)))

# 
pred = rf.predict(X_test_final)
pred_ds = test_df[['PassengerId']]
pred_ds['Survived'] = pred
pred_ds.head()
pred_ds.to_csv('submission3.csv', index=False)
# Ada Boost.
from sklearn.model_selection import cross_val_score

from sklearn.ensemble import AdaBoostClassifier

ab = AdaBoostClassifier(n_estimators=500)

ab.fit(X_train, y_train)
print('Accuracy of Ada Boost classifier on training set: {:.2f}'
     .format(ab.score(X_train, y_train)))
print('Accuracy of Ada Boost classifier on test set: {:.2f}'
     .format(ab.score(X_test, y_test)))
# XGBoost
from xgboost import XGBClassifier
xb = XGBClassifier(n_estimators=500)
xb.fit(X_train, y_train)
print('Accuracy of XG Boost classifier on training set: {:.2f}'
     .format(xb.score(X_train, y_train)))
print('Accuracy of XG Boost classifier on test set: {:.2f}'
     .format(xb.score(X_test, y_test)))

pred = xb.predict(X_test_final)
pred_ds = test_df[['PassengerId']]
pred_ds['Survived'] = pred
pred_ds.head()
pred_ds.to_csv('submission4.csv', index=False)
# Plot Confusion Matrix
import itertools

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
# Confusion Matrix for KNN
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
pred = knn.predict(X_test)

cnf_matrix = confusion_matrix(y_test, pred)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=[0,1],
                      title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=[0,1], normalize=True,
                      title='Normalized confusion matrix')

plt.show()

print(cnf_matrix)
print(classification_report(y_test, pred))
