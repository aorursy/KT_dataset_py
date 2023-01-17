# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import numpy as np, pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.pipeline import Pipeline

from sklearn.svm import SVC

from sklearn.model_selection import GridSearchCV
train = pd.read_csv('../input/train.csv', index_col = False)
test = pd.read_csv('../input/test.csv', index_col = False)
print(train.shape)

print(test.shape)
train.head()
test.head()
sns.countplot(train['label'])

plt.show()
X_ = train.iloc[:, 1:]

y_ = train.iloc[:, 0]

print(X_.shape)

print(y_.shape)
X_train, X_validate, y_train, y_validate = train_test_split(X_, y_, test_size = 0.8, random_state = 30, stratify = y_)
print(X_train.shape)

print(X_validate.shape)

print(y_train.shape)

print(y_validate.shape)
steps = [('scaler', StandardScaler()), ('SVM', SVC(kernel = 'poly'))]

pipeline = Pipeline(steps)
parameters = {'SVM__C': [0.001, 0.1, 10, 100, 10e5], 

              'SVM__gamma': [0.01, 0.1, 10, 100]}
grid = GridSearchCV(pipeline, 

                    param_grid = parameters, 

                    cv = 5, 

                    n_jobs = -1, 

                    scoring = 'accuracy', 

                    verbose = 1, 

                    return_train_score = True)
grid.fit(X_train, y_train)
cv_results = pd.DataFrame(grid.cv_results_)

cv_results.head()
cv_results['param_SVM__C'] = cv_results['param_SVM__C'].astype('int')



plt.figure(figsize = (16,6))



plt.subplot(221)

gamma_01 = cv_results[cv_results['param_SVM__gamma'] == 0.01]



plt.plot(gamma_01['param_SVM__C'], gamma_01['mean_test_score'])

plt.plot(gamma_01['param_SVM__C'], gamma_01['mean_train_score'])

plt.xlabel('C')

plt.ylabel('Accuracy')

plt.title('Gamma = 0.01')

plt.legend(['test accuracy', 'train accuracy'], loc = 'upper left')

plt.xscale('log')



plt.subplot(222)

gamma_1 = cv_results[cv_results['param_SVM__gamma'] == 0.1]



plt.plot(gamma_1['param_SVM__C'], gamma_1['mean_test_score'])

plt.plot(gamma_1['param_SVM__C'], gamma_1['mean_train_score'])

plt.xlabel('C')

plt.ylabel('Accuracy')

plt.title('Gamma = 0.1')

plt.legend(['test accuracy', 'train accuracy'], loc = 'upper left')

plt.xscale('log')



plt.subplot(223)

gamma_10 = cv_results[cv_results['param_SVM__gamma'] == 10]



plt.plot(gamma_10['param_SVM__C'], gamma_10['mean_test_score'])

plt.plot(gamma_10['param_SVM__C'], gamma_10['mean_train_score'])

plt.xlabel('C')

plt.ylabel('Accuracy')

plt.title('Gamma = 10')

plt.legend(['test accuracy', 'train accuracy'], loc = 'upper left')

plt.xscale('log')



plt.subplot(224)

gamma_100 = cv_results[cv_results['param_SVM__gamma'] == 100]



plt.plot(gamma_100['param_SVM__C'], gamma_100['mean_test_score'])

plt.plot(gamma_100['param_SVM__C'], gamma_100['mean_train_score'])

plt.xlabel('C')

plt.ylabel('Accuracy')

plt.title('Gamma = 100')

plt.legend(['test accuracy', 'train accuracy'], loc = 'upper left')

plt.xscale('log')



plt.show()
print('Best Score', grid.best_score_)

print('Best hyperparameters', grid.best_params_)
y_pred_ = model.predict(X_train)

print('Accuracy', metrics.accuracy_score(y_train, y_pred_))
model = SVC(C = 0.001, gamma = 0.1, kernel = 'poly')

model.fit(X_train, y_train)

y_pred = model.predict(X_validate)
from sklearn import metrics

print('Accuracy', metrics.accuracy_score(y_validate, y_pred))
y_pred_t = model.predict(test)

y_pred_t
len(y_pred_t)