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
train = pd.read_csv('../input/training.csv', delimiter = ";")

val = pd.read_csv('../input/validation.csv', delimiter = ";")
train.head()
train.describe()
train.info()
total = pd.concat([train, val], axis=0)
total.shape

train.shape

val.shape
total['variable2'] = total['variable2'].str.replace(',','.').astype(np.float64)

total['variable3'] = total['variable3'].str.replace(',','.').astype(np.float64)

total['variable8'] = total['variable8'].str.replace(',','.').astype(np.float64)
total[:train.shape[0]].info()
total.head()
total.drop('variable18', axis = 1, inplace = True)
total[['variable2']].boxplot()
total['variable2'] = total['variable2'].fillna(total['variable2'].median())

total['variable3'] = total['variable3'].fillna(total['variable3'].median())

total['variable8'] = total['variable8'].fillna(total['variable8'].median())

total['variable14'] = total['variable14'].fillna(total['variable14'].median())

total['variable17'] = total['variable17'].fillna(total['variable17'].median())
total.info()
total.columns
for i in total.columns:

    if total[i].dtypes == "O":

        total[i] = total[i].fillna(total[i].mode()[0])
total.info()
total.head()
total['classLabel'].unique()
total[total['classLabel'] ==  'no.'].shape

sampling_data = total[total['classLabel'] ==  'no.']
total = pd.concat([total, sampling_data, sampling_data, sampling_data], axis=0)
X = pd.get_dummies(total.iloc[:, total.columns != 'classLabel'])
X_train = X[ :train.shape[0] + (2 * sampling_data.shape[0])]

X_test = X[train.shape[0]: ]

y_train = total['classLabel'][ :train.shape[0]+ (2 * sampling_data.shape[0])]

y_test = total['classLabel'][train.shape[0]: ]
X_train.shape ,  X_test.shape , y_train.shape , y_test.shape
y_train = y_train.values.reshape(-1, 1)

y_test = y_test.values.reshape(-1, 1)
from sklearn.preprocessing import MinMaxScaler

sc_X = MinMaxScaler()

X_train_sc = sc_X.fit_transform(X_train)

X_test_sc = sc_X.transform(X_test)
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from sklearn.model_selection import cross_val_score

# machine learning

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.tree import DecisionTreeClassifier
logi_clf = LogisticRegression(solver='lbfgs', max_iter=500)

logi_parm = {"C": [0.1, 0.5, 1, 5, 10, 50],

            'random_state': [0,1,2,3,4,5]}



svm_clf = SVC(probability=True)

svm_parm = {'kernel': ['rbf', 'poly'], 

            'C': [1, 5, 50, 100, 500, 1000], 

            'degree': [3, 5, 7], 

            'gamma': ['auto', 'scale'],

           'random_state': [0,1,2,3,4,5]}



dt_clf = DecisionTreeClassifier()

dt_parm = {'criterion':['gini', 'entropy'],

          'random_state': [0,1,2,3,4,5]}



knn_clf = KNeighborsClassifier()

knn_parm = {'n_neighbors':[5, 10, 15, 20], 

            'weights':['uniform', 'distance'], 

            'p': [1,2]}



gnb_clf = GaussianNB()

gnb_parm = {'priors':['None']}



clfs = [logi_clf, svm_clf, dt_clf, knn_clf]

params = [logi_parm, svm_parm, dt_parm, knn_parm] 

clf_names = ['logistic', 'SVM', 'DT', 'KNN', 'GNB']
y_train
import sklearn.ensemble as ens



clf = ens.RandomForestClassifier()

param = {'n_estimators':[10,50,100,500,100],

         'criterion': ['gini', 'entropy'],}

RF = RandomizedSearchCV(clf, param, cv=5)

RF.fit(X_train_sc, y_train)

RF.best_score_
from sklearn.metrics import accuracy_score, confusion_matrix
accuracy_score(y_test, RF.predict(X_test))
import xgboost as xgb



xgb_model = xgb.XGBClassifier(objective="binary:logistic", random_state=42)

xgb_model.fit(X_train_sc, y_train)



y_pred = xgb_model.predict(X_test_sc)



print(confusion_matrix(y_test, y_pred))
accuracy_score(y_test, y_pred)