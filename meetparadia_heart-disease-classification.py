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
import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import warnings

from sklearn.preprocessing import MinMaxScaler, scale, StandardScaler

from sklearn.model_selection import train_test_split, cross_val_score

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import make_scorer, precision_score

from sklearn.linear_model import LogisticRegression

from sklearn.svm import LinearSVC

from sklearn import svm

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import confusion_matrix

warnings.filterwarnings('ignore')

%matplotlib inline
data = pd.read_csv('/kaggle/input/heart-disease-uci/heart.csv')
data.head()
data.info()
data.describe()
sns.pairplot(data)
# Feature correlation heatmap

fig, ax = plt.subplots(1, 1, figsize=(10, 10))

corr = data.drop(columns=['target']).corr()

corr = corr.round(decimals=2)

corr = corr.where(np.tril(np.ones(corr.shape)).astype(np.bool)) # make heatmap lower triangular (remove redundant info)

sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns, annot=True, ax=ax, cmap = 'coolwarm')

plt.xticks(rotation=90)

plt.yticks(rotation=0)

ax.set_ylim(len(corr)+0.5, -0.5)

plt.show()
# Histogram for all features



fig, ax = plt.subplots(1, 1, figsize=(15, 15))

data.drop(columns=['target']).hist(ax=ax)

plt.show()
sns.countplot(data['sex'], hue=data['target'])
sns.countplot(data['cp'], hue=data['target'])
sns.scatterplot(x=data['age'], y=data['chol'], hue=data['target'])
sns.countplot(data['thal'], hue=data['target'])
cp = pd.get_dummies(data['cp'], drop_first=True)

restecg = pd.get_dummies(data['restecg'], drop_first=True)

slope = pd.get_dummies(data['slope'], drop_first=True)

ca = pd.get_dummies(data['ca'], drop_first=True)
cp.columns = ['cp_1', 'cp_2', 'cp_3']

restecg.columns = ['restecg_1', 'restecg_2']

slope.columns = ['slope_1', 'slope_2']

ca.columns = ['ca_1', 'ca_2', 'ca_3', 'ca_4']
data.drop(['cp', 'restecg', 'slope', 'ca'], axis=1, inplace=True)
data = pd.concat([data, cp, restecg, slope,ca], axis=1)
data.head()
X = data.drop(['target'], axis=1)

y = data['target']



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.fit_transform(X_test)
train_score_array = []

test_score_array = []



for k in range(1, 20):

    knn = KNeighborsClassifier(k)

    knn.fit(X_train, y_train)

    train_score_array.append(knn.score(X_train, y_train))

    test_score_array.append(knn.score(X_test, y_test))

    

x_axis = range(1, 20)

plt.subplots(figsize = (20, 5))

plt.plot(train_score_array, label='Train score array', c='g')

plt.plot(test_score_array, label='Test score array', c='b')

plt.xlabel('N neighbors')

plt.ylabel('Accuracy')

plt.xticks(x_axis, np.arange(20))

plt.grid()

plt.legend()

plt.show()
param_grid = {'n_neighbors' : np.arange(1, 20)}

grid_search = GridSearchCV(knn, param_grid, cv=5, scoring= 'precision_macro' , return_train_score=True)

grid_search.fit(X_train, y_train)



grid_search.best_params_
knn = KNeighborsClassifier(n_neighbors=11)

knn.fit(X_train, y_train)

print('Training score:', knn.score(X_train, y_train))

print('Testing score:', knn.score(X_test, y_test))
y_pred_train = knn.predict(X_train)

y_pred_test = knn.predict(X_test)



knn_train_precision_score = precision_score(y_train, y_pred_train, average='macro')

knn_test_precision_score = precision_score(y_test, y_pred_test, average='macro')



print('Train Precision score:', knn_train_precision_score)

print('Test Precision score:', knn_test_precision_score)

confusion_matrix(y_test, y_pred_test)
knn = KNeighborsClassifier(n_neighbors=11)

knn.fit(X_train, y_train)

train_score = cross_val_score(knn, X_train, y_train)

test_score = cross_val_score(knn, X_test, y_test)

print('Cross-validation scores:', train_score)

print('Cross-validation scores:', test_score)

print('Average Train score:', train_score.mean())

print('Average Test score:', test_score.mean())
c_range = [0.001, 0.01, 0.1, 1, 10, 100, 1000]

train_score_l1 = []

train_score_l2 = []

test_score_l1 = []

test_score_l2 = []



for c in c_range:

    log_l1 = LogisticRegression(penalty = 'l1', C = c, solver = 'liblinear', max_iter = 500)

    log_l2 = LogisticRegression(penalty = 'l2', C = c, solver = 'lbfgs', max_iter = 500)

    log_l1.fit(X_train, y_train)

    log_l2.fit(X_train, y_train)

    train_score_l1.append(log_l1.score(X_train, y_train))

    train_score_l2.append(log_l2.score(X_train, y_train))

    test_score_l1.append(log_l1.score(X_test, y_test))

    test_score_l2.append(log_l2.score(X_test, y_test))

    

plt.subplots(figsize = (20,5))

plt.plot(c_range, train_score_l1, label = 'Train score, penalty = l1')

plt.plot(c_range, test_score_l1, label = 'Test score, penalty = l1')

plt.plot(c_range, train_score_l2, label = 'Train score, penalty = l2')

plt.plot(c_range, test_score_l2, label = 'Test score, penalty = l2')

plt.legend()

plt.xlabel('Regularization parameter: C')

plt.ylabel('Accuracy')

plt.xscale('log')
lreg_clf = LogisticRegression()



param_grid = {'C': [0.00001,0.0001,0.001,0.01,0.1,1,10,100], 'penalty': ['l1', 'l2']}



grid_search = GridSearchCV(lreg_clf, param_grid, cv=5, scoring= 'precision_macro' ,return_train_score=True)

grid_search.fit(X_train, y_train)



grid_search.best_params_
lreg_clf = LogisticRegression(C=10, penalty= 'l2')

lreg_clf.fit(X_train, y_train)



y_pred_train = lreg_clf.predict(X_train)

y_pred_test = lreg_clf.predict(X_test)



lreg_train_precision_score = precision_score(y_train, y_pred_train, average='macro')

lreg_test_precision_score = precision_score(y_test, y_pred_test, average='macro')



print('Train Precision score:', lreg_train_precision_score)

print('Test Precision score:', lreg_test_precision_score)



confusion_matrix(y_test, y_pred_test)
lreg_clf = LogisticRegression(C=10, penalty= 'l2')

lreg_clf.fit(X_train, y_train)

train_score = cross_val_score(lreg_clf, X_train, y_train)

test_score = cross_val_score(lreg_clf, X_test, y_test)

print('Cross-validation scores:', train_score)

print('Cross-validation scores:', test_score)

print('Average Train score:', train_score.mean())

print('Average Test score:', test_score.mean())
LSVC_clf = LinearSVC()



param_grid = {'C': [0.00001,0.0001,0.001,0.01,0.1,1,10,100]}



grid_search = GridSearchCV(LSVC_clf, param_grid, cv=5, scoring='precision_macro', return_train_score=True, iid=False)

grid_search.fit(X_train, y_train)



grid_search.best_params_
LSVC_clf = LinearSVC(C=0.1)

LSVC_clf.fit(X_train, y_train)



y_pred_train = LSVC_clf.predict(X_train)

y_pred_test = LSVC_clf.predict(X_test)



LSVC_train_precision_score = precision_score(y_train, y_pred_train, average='macro')

LSVC_test_precision_score = precision_score(y_test, y_pred_test, average='macro')



print('Train Precision score:', LSVC_train_precision_score)

print('Test Precision score:', LSVC_test_precision_score)

confusion_matrix(y_test, y_pred_test)
LSVC_clf = LinearSVC(C=0.1)

LSVC_clf.fit(X_train, y_train)

train_score = cross_val_score(LSVC_clf, X_train, y_train)

test_score = cross_val_score(LSVC_clf, X_test, y_test)

print('Cross-validation scores:', train_score)

print('Cross-validation scores:', test_score)

print('Average Train score:', train_score.mean())

print('Average Test score:', test_score.mean())
KSVC_clf = svm.SVC(kernel='rbf', random_state=0)



param_grid = {'C': [0.0001,0.001,0.01,0.1,1,10],

          'gamma': [0.0001,0.001,0.1,1,10]}



grid_search = GridSearchCV(KSVC_clf, param_grid, cv=5, scoring= 'precision_macro', return_train_score=True, iid=False)

grid_search.fit(X_train, y_train)



grid_search.best_params_
KSVC_clf = svm.SVC(kernel='rbf', C=10, gamma=0.001, probability=True)

KSVC_clf.fit(X_train, y_train)



y_pred_train = KSVC_clf.predict(X_train)

y_pred_test = KSVC_clf.predict(X_test)



KSVC_train_precision_score = precision_score(y_train, y_pred_train, average='macro')

KSVC_test_precision_score = precision_score(y_test, y_pred_test, average='macro')



print('Train Precision score:', KSVC_train_precision_score)

print('Test Precision score:', KSVC_test_precision_score)

confusion_matrix(y_test, y_pred_test)
KSVC_clf = svm.SVC(kernel='rbf',C=10, gamma=0.001, probability=True)

KSVC_clf.fit(X_train, y_train)

train_score = cross_val_score(KSVC_clf, X_train, y_train)

test_score = cross_val_score(KSVC_clf, X_test, y_test)

print('Cross-validation scores:', train_score)

print('Cross-validation scores:', test_score)

print('Average Train score:', train_score.mean())

print('Average Test score:', test_score.mean())
dt_clf = DecisionTreeClassifier()

param_grid = {'max_depth': np.arange(1,20)}



grid_search = GridSearchCV(dt_clf, param_grid, return_train_score=True, scoring='precision_macro', iid=False)

grid_search.fit(X_train, y_train)



grid_search.best_params_
dt_clf = DecisionTreeClassifier(max_depth=3)

dt_clf.fit(X_train, y_train)



y_pred_train = dt_clf.predict(X_train)

y_pred_test = dt_clf.predict(X_test)



dt_train_precision_score = precision_score(y_train, y_pred_train, average='macro')

dt_test_precision_score = precision_score(y_test, y_pred_test, average='macro')



print('Train Precision score:', dt_train_precision_score)

print('Test Precision score:', dt_test_precision_score)

confusion_matrix(y_test, y_pred_test)
dt_clf = DecisionTreeClassifier(max_depth=3)

dt_clf.fit(X_train, y_train)

train_score = cross_val_score(dt_clf, X_train, y_train)

test_score = cross_val_score(dt_clf, X_test, y_test)

print('Cross-validation scores:', train_score)

print('Cross-validation scores:', test_score)

print('Average Train score:', train_score.mean())

print('Average Test score:', test_score.mean())