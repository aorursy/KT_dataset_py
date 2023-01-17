#importing modules

import matplotlib.pyplot as plt

import pandas as pd

import numpy as np

%matplotlib inline

import warnings

warnings.filterwarnings('ignore')

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.

data = pd.read_csv('../input/diabetes.csv')

data.head()
data.tail()
data.shape
data.describe()
data.groupby('Outcome').size()
# Visualizing datas

data.hist(figsize=(16,14))
data.plot(kind= 'box' , subplots=True, layout=(3,3), sharex=False, sharey=False, figsize=(16,14))
column_x = data.columns[0:len(data.columns) - 1]

column_x
corr = data[data.columns].corr()

corr
#Extracting the important features for better accuracy

from sklearn.feature_selection import SelectKBest

from sklearn.feature_selection import chi2

X = data.iloc[:,0:8]

Y = data.iloc[:,8]

select_top_4 = SelectKBest(score_func=chi2, k = 5)
fit = select_top_4.fit(X,Y)

features = fit.transform(X)
features[0:5]
data.head()
features = ['Pregnancies', 'Glucose', 'Insulin', 'BMI', 'Age']

req_features = data[features]

req_features.head()
req_outcome = data['Outcome']

req_outcome.head()
#Splitting data for train and test

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(req_features, req_outcome, test_size=.25, random_state = 12)
#Decision Tree

from sklearn import tree

t_clf = tree.DecisionTreeClassifier()

t_clf = t_clf.fit(X_train, Y_train)

t_acc = t_clf.score(X_test, Y_test)

print (t_acc)
# SVM with linear kernel

from sklearn import svm

s_clf = svm.SVC(kernel='linear')

s_clf = s_clf.fit(X_train, Y_train)

s_acc = s_clf.score(X_test, Y_test)

print (s_acc)
#SVM with rbf kernel

from sklearn import svm

s_clf = svm.SVC(kernel='rbf')

s_clf = s_clf.fit(X_train, Y_train)

s_acc = s_clf.score(X_test, Y_test)

print (s_acc)
#Naive Bayes

from sklearn.naive_bayes import GaussianNB

n_clf = GaussianNB()

n_clf = n_clf.fit(X_train, Y_train)

n_acc = n_clf.score(X_test, Y_test)

print (n_acc)