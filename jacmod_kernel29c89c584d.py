import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


# visualize
from matplotlib import pyplot as plt
import seaborn as sns

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

data = pd.read_csv('/kaggle/input/heart-disease-uci/heart.csv')
df = data.copy()
df.head()

df.describe()
df.info()
df[['sex', 'target']].groupby(['sex'], as_index=False).mean().sort_values(by='target', ascending=False)
df[['cp', 'target']].groupby(['cp'], as_index=False).mean().sort_values(by='target', ascending=False)
df[['fbs', 'target']].groupby(['fbs'], as_index=False).mean().sort_values(by='target', ascending=False)
df[['restecg', 'target']].groupby(['restecg'], as_index=False).mean().sort_values(by='target', ascending=False)
df[['exang', 'target']].groupby(['exang'], as_index=False).mean().sort_values(by='target', ascending=False)
df[['slope', 'target']].groupby(['slope'], as_index=False).mean().sort_values(by='target', ascending=False)
df[['ca', 'target']].groupby(['ca'], as_index=False).mean().sort_values(by='target', ascending=False)
df[['thal', 'target']].groupby(['thal'], as_index=False).mean().sort_values(by='target', ascending=False)
age_hist = sns.FacetGrid(df, col = 'target')
age_hist.map(plt.hist, 'age', bins = 25)
trest_hist = sns.FacetGrid(df, col = 'target')
trest_hist.map(plt.hist, 'trestbps', bins = 25)
chol_hist = sns.FacetGrid(df, col = 'target')
chol_hist.map(plt.hist, 'chol', bins = 25)
thalach_hist = sns.FacetGrid(df, col = 'target')
thalach_hist.map(plt.hist, 'thalach', bins = 25)
oldpeak_hist = sns.FacetGrid(df, col = 'target')
oldpeak_hist.map(plt.hist, 'oldpeak', bins = 25)
age_sex = sns.FacetGrid(df, col = 'target', row = 'sex')
age_sex.map(plt.hist, 'age', bins = 25)
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

from sklearn.model_selection import GridSearchCV

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
X_train, X_test, y_train, y_test = train_test_split(df.drop(['target'], axis = 1), df['target'], test_size = 0.25, stratify = df['target'])
X_test
scaler = MinMaxScaler()

scaler.fit(X_train)
scaler.transform(X_train)
scaler.transform(X_test)
accuracy_list = [] # create list of accuracies on test data
recall_list = [] # create list of recalls on test data
param_grid = { 
    'C' : [0.01, 0.1, 1, 10, 100],
    'gamma' :  [0.01, 0.1, 1, 10],
    'kernel' : ['rbf', 'sigmoid']}

sv = SVC()
clf_sv = GridSearchCV(sv, param_grid, n_jobs=-1)
clf_sv.fit(X_train, y_train)
print(clf_sv.best_score_)
print(clf_sv.best_estimator_)
final_svm = SVC(C = 10, gamma = 0.01, kernel = 'rbf')
final_svm.fit(X_train, y_train)
svm_pred = final_svm.predict(X_test)
accuracy_list.append(accuracy_score(y_test, svm_pred))
svm_cm = confusion_matrix(y_test, svm_pred)
recall_list.append(svm_cm[1][1] / (svm_cm[1][1] + svm_cm[1][0]))
param_grid = { 
    'max_depth': [10, 30, 50, 70], 
    "min_samples_leaf" : [1, 5, 10, 50], 
    "min_samples_split" : [2, 4, 10, 12, 16, 18, 25],
    'n_estimators': [5, 10, 50, 100, 120, 140]}

rfc = RandomForestClassifier()
clf_rfc = GridSearchCV(rfc, param_grid, n_jobs=-1)
clf_rfc.fit(X_train, y_train)
print(clf_rfc.best_score_)
print(clf_rfc.best_estimator_)
final_rfc = RandomForestClassifier(max_depth = 30, min_samples_leaf = 1, min_samples_split = 16, n_estimators = 10)
final_rfc.fit(X_train, y_train)
rfc_pred = final_rfc.predict(X_test)
accuracy_list.append(accuracy_score(y_test, rfc_pred))
rfc_cm = confusion_matrix(y_test, rfc_pred)
recall_list.append(rfc_cm[1][1] / (rfc_cm[1][1] + rfc_cm[1][0]))
param_grid = { 
    "criterion" : ["gini", "entropy"], 
    "min_samples_leaf" : [1, 5, 10, 25, 50, 70], 
    "min_samples_split" : [2, 4, 10, 12, 16, 18, 25, 35]}

dt = DecisionTreeClassifier()
clf_dt = GridSearchCV(dt, param_grid, n_jobs=-1)
clf_dt.fit(X_train, y_train)
print(clf_dt.best_score_)
print(clf_dt.best_estimator_)
final_dt = DecisionTreeClassifier(criterion = 'gini', min_samples_leaf = 10, min_samples_split = 25)
final_dt.fit(X_train, y_train)
dt_pred = final_rfc.predict(X_test)
accuracy_list.append(accuracy_score(y_test, dt_pred))
dt_cm = confusion_matrix(y_test, dt_pred)
recall_list.append(dt_cm[1][1] / (dt_cm[1][1] + dt_cm[1][0]))
param_grid = { 
    'penalty': ['l1', 'l2'], 
    "C" : [1, 0.1, 0.001, 0.0001], 
    "solver" : ['lbfgs', 'liblinear']}

lr = LogisticRegression()
clf_lr = GridSearchCV(lr, param_grid, n_jobs=-1)
clf_lr.fit(X_train, y_train)
print(clf_lr.best_score_)
print(clf_lr.best_estimator_)
final_lr = LogisticRegression(C = 1, penalty = 'l2', solver = 'lbfgs')
final_lr.fit(X_train, y_train)
lr_pred = final_lr.predict(X_test)
accuracy_list.append(accuracy_score(y_test, lr_pred))
lr_cm = confusion_matrix(y_test, lr_pred)
recall_list.append(lr_cm[1][1] / (lr_cm[1][1] + lr_cm[1][0]))
gnb = GaussianNB()
gnb.fit(X_train, y_train)
gnb_pred = gnb.predict(X_test)
accuracy_list.append(accuracy_score(y_test, gnb_pred))
gnb_cm = confusion_matrix(y_test, gnb_pred)
recall_list.append(gnb_cm[1][1] / (gnb_cm[1][1] + gnb_cm[1][0]))
models = ['SVM', 'Random Forest', 'Decision Tree', 'Logistic Regression', 'Naive Bayes']
fig = plt.figure()
ax = fig.add_axes([0,0,1.25,1])
ax.bar(models, accuracy_list)
plt.title('Accuracy scores')
plt.show()
fig_recall = plt.figure()
ax = fig_recall.add_axes([0,0,1.25,1])
ax.bar(models, recall_list)
plt.title('Recall scores')
plt.show()