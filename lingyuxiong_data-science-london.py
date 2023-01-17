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
import pandas as pd

import numpy as np

from sklearn.model_selection import cross_val_score

from sklearn.metrics import accuracy_score 
train = pd.read_csv('../input/data-science-london-scikit-learn/train.csv', header=None)

train_label = pd.read_csv('../input/data-science-london-scikit-learn/trainLabels.csv', header=None)

test = pd.read_csv('../input/data-science-london-scikit-learn/test.csv', header=None)
train.head()
train.shape, train_label.shape, test.shape
train.describe()
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(train, train_label, test_size=0.30, random_state=101)

x_train.shape, x_test.shape, y_train.shape, y_test.shape
from sklearn.neighbors import KNeighborsClassifier

knn_model = KNeighborsClassifier()

knn_model.fit(x_train, y_train.values.ravel())

predicted = knn_model.predict(x_test)

print('KNN',accuracy_score(y_test, predicted))
from sklearn.tree import DecisionTreeClassifier

dtree_model = DecisionTreeClassifier()

dtree_model.fit(x_train, y_train.values.ravel())

predicted = dtree_model.predict(x_test)

print('Decision Tree', accuracy_score(y_test, predicted))
from sklearn.ensemble import RandomForestClassifier

rfc_model = RandomForestClassifier()

rfc_model.fit(x_train, y_train.values.ravel())

predicted = rfc_model.predict(x_test)

print('Random Forest', accuracy_score(y_test, predicted))
from sklearn.svm import SVC

svc_model = SVC(gamma='auto')

svc_model.fit(x_train, y_train.values.ravel())

predicted = svc_model.predict(x_test)

print('SVM', accuracy_score(y_test, predicted))
from xgboost import XGBClassifier

xgb_model = XGBClassifier()

xgb_model.fit(x_train, y_train.values.ravel())

predicted = xgb_model.predict(x_test)

print('XGBoost', accuracy_score(y_test, predicted))
from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import GridSearchCV, cross_val_score

from sklearn.mixture import GaussianMixture

from sklearn.svm import SVC
x_all = np.r_[train,test]

print('x_all shape :',x_all.shape)



# USING THE GAUSSIAN MIXTURE MODEL 

lowest_bic = np.infty

bic = []

n_components_range = range(1, 7)

cv_types = ['spherical', 'tied', 'diag', 'full']

for cv_type in cv_types:

    for n_components in n_components_range:

        gmm = GaussianMixture(n_components=n_components,covariance_type=cv_type)

        gmm.fit(x_all)

        bic.append(gmm.aic(x_all))

        if bic[-1] < lowest_bic:

            lowest_bic = bic[-1]

            best_gmm = gmm

            

best_gmm.fit(x_all)

gmm_train = best_gmm.predict_proba(train)

gmm_test = best_gmm.predict_proba(test)
rfc = RandomForestClassifier()

n_estimators = [10, 50, 100, 200,400]

max_depth = [3, 10, 20, 40]

param_grid = dict(n_estimators=n_estimators,max_depth=max_depth)

grid_search_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, scoring='accuracy', n_jobs=-1).fit(gmm_train, train_label.values.ravel())

rfc_best = grid_search_rfc.best_estimator_

print('Random Forest', grid_search_rfc.best_score_)
rfc_best.fit(gmm_train,train_label.values.ravel())

pred  = rfc_best.predict(gmm_test)

rfc_best_pred = pd.DataFrame(pred)



rfc_best_pred.index += 1



rfc_best_pred.columns = ['Solution']

rfc_best_pred['Id'] = np.arange(1,rfc_best_pred.shape[0]+1)

rfc_best_pred = rfc_best_pred[['Id', 'Solution']]



rfc_best_pred.to_csv('Submission_GMM_RFC.csv',index=False)