# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis, LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, recall_score
df = pd.read_csv('/kaggle/input/mines-vs-rocks/sonar.all-data.csv', header=None)
df
df.describe().T
df.info()
fig, axs = plt.subplots(figsize=(10, 8))
sns.countplot(df[60], ax=axs)
plt.show()
df.hist(figsize=(15, 10))
plt.figure(figsize=(15, 10))
sns.heatmap(df.corr())
X = df.drop(columns=60).values
y = df[60]
y = y.map({'R' : 0, 'M' : 1}).values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
classifiers = [('KNN', KNeighborsClassifier()), 
               ('SVC', SVC()), 
               ('GPC', GaussianProcessClassifier()), 
               ('DTC', DecisionTreeClassifier()), 
               ('RFC', RandomForestClassifier()), 
               ('MLPC', MLPClassifier()), 
               ('ABC', AdaBoostClassifier()), 
               ('GNB', GaussianNB()), 
               ('QDA', QuadraticDiscriminantAnalysis()), 
               ('LDA', LinearDiscriminantAnalysis()), 
               ('LR', LogisticRegression())]
results = []
names = []
scoring = 'accuracy'

for name, classifier in classifiers:
    kfold = KFold(n_splits=10, shuffle=True)
    cv_score = cross_val_score(estimator=classifier, X=X_train, y=y_train, scoring=scoring)
    results.append(cv_score)
    names.append(name)
    print('Classifier: {}, Mean Accuracy: {}, StDev: {}'.format(name, cv_score.mean(), cv_score.std()))
fig, axs = plt.subplots(figsize=(15, 10))
axs.boxplot(results)
axs.set_xticklabels(names, fontdict={'size' : 18})
plt.show()
scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
results_scaled_data = []
names = []

for name, model in classifiers:
    kfold = KFold(n_splits=10, shuffle=True)
    cv_score = cross_val_score(estimator=model, X=X_train_scaled, y=y_train, scoring='accuracy')
    results_scaled_data.append(cv_score)
    names.append(name)
    print('Model: {}, Mean Score: {}, Score StDev: {}'.format(name, cv_score.mean(), cv_score.std()))
fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(16, 12), sharex=True, sharey=True)
axs[0].boxplot(results_scaled_data)
axs[0].set_title('Standardised Data', fontdict={'size' : 18})
axs[0].set_xticklabels(names)
axs[0].grid(linestyle='dashed', linewidth=0.2, color='black')

axs[1].boxplot(results)
axs[1].set_title('Non-Standardised Data', fontdict={'size' : 18})
axs[1].grid(linestyle='dashed', linewidth=0.2, color='black')
svc = SVC()

param_grid = {'C' : [1, 10, 100, 1000], 'kernel' : ['linear', 'rbf', 'sigmoid'], 
              'gamma' : ['scale', 'auto', 1.0, 0.1, 0.01, 0.001, 0.0001]}

kfold = KFold(n_splits=10, shuffle=True)
gsearch = GridSearchCV(estimator=svc, param_grid=param_grid, scoring='accuracy', cv=kfold)
gsearch.fit(X_train_scaled, y_train)
gsearch.best_params_
svc = SVC(**gsearch.best_params_)
svc.fit(X_train_scaled, y_train)
svc_preds = svc.predict(X_test_scaled)
svc_cm = confusion_matrix(y_test, svc_preds)

sns.heatmap(svc_cm, annot=True)
svc_clf_report = classification_report(y_test, svc_preds)
print(svc_clf_report)
mlpc = MLPClassifier()

mlpc_grid = {'hidden_layer_sizes' : [(100,), (100, 100), (150, 150), (200, 200)], 
            'alpha' : [0.0001, 0.001, 0.01, 0.1], 
            'learning_rate' : ['constant', 'invscaling', 'adaptive']}

mlpc_kfold = KFold(n_splits=10, shuffle=True)
mlpc_search = GridSearchCV(estimator=mlpc, param_grid=mlpc_grid, scoring='accuracy', cv=mlpc_kfold)
mlpc_search.fit(X_train_scaled, y_train)
mlpc_search.best_params_
mlp_clf = MLPClassifier(**mlpc_search.best_params_)
mlp_clf.fit(X_train_scaled, y_train)
mlpc_preds = mlp_clf.predict(X_test_scaled)
mlpc_cm = confusion_matrix(y_test, mlpc_preds)
sns.heatmap(mlpc_cm, annot=True)
mlpc_clf_report = classification_report(y_test, mlpc_preds)
print(mlpc_clf_report)
gpc = GaussianProcessClassifier()

gpc_param_grid = {'optimizer' : ['fmin_l_bfgs_b', None], 'n_restarts_optimizer' : [0, 1, 3, 5, 9], 
                 'max_iter_predict' : [50, 100, 200, 500, 1000]}

gpc_kfold = KFold(n_splits=10, shuffle=True)
gpc_grid_search = GridSearchCV(estimator=gpc, param_grid=gpc_param_grid, scoring='accuracy', cv=gpc_kfold)
gpc_grid_search.fit(X_train_scaled, y_train)
print(gpc_grid_search.best_params_)
gp_clf = GaussianProcessClassifier(**gpc_grid_search.best_params_)
gp_clf.fit(X_train_scaled, y_train)
gpc_preds = gp_clf.predict(X_test_scaled)
gpc_cm = confusion_matrix(y_test, gpc_preds)
sns.heatmap(gpc_cm, annot=True)
gpc_cr = classification_report(y_test, gpc_preds)
print(gpc_cr)
print('SVC recall score: {}'.format(recall_score(y_test, svc_preds)))
print('MLPClassifier recall score: {}'.format(recall_score(y_test, mlpc_preds)))
print('GaussianProcessClassifier recall score: {}'.format(recall_score(y_test, gpc_preds)))