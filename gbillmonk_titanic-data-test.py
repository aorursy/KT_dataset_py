# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')
def rectify_age(df):
    df.Age = df.Age.fillna(-0.5)
    bins = (-1, 0, 5, 12, 18, 25, 40, 60, 100)
    group_names = ['unknown', 'baby', 'kid', 'teen', 'yadult', 'adult', 'ysenior', 'senior']
    categories = pd.cut(df.Age, bins, labels=group_names)
    df.Age = categories
    return df

def fare_bucket(df):
    df.Fare = df.Fare.fillna(-0.5)
    bins = (-1, 0, 100, 200, 300, 400, 500, 600)
    bins = (-1, 0, 20, 30, 40, 60, 100, 600)
    group_names = ['Unknown', 'first', 'second', 'third', 'fourth', 'fifth', 'sixth']
    categories = pd.cut(df.Fare, bins, labels=group_names)
    df.Fare = categories
    return df

def transform(df):
    df = rectify_age(df)
    df = fare_bucket(df)
    return df

train_df = transform(train_df)
test_df = transform(test_df)

#sns.barplot(x='Age', data=train_df)
train_df = train_df.drop(['PassengerId','Name', 'Ticket', 'Cabin', 'Embarked'],axis=1)
test_df = test_df.drop(['PassengerId','Name', 'Ticket', 'Cabin', 'Embarked'],axis=1)

from sklearn import preprocessing
def encode_features(df_train, df_test):
    features = ['Sex', 'Age', 'Fare']
    df_combined = pd.concat([df_train[features], df_test[features]])
    
    for feature in features:
        le = preprocessing.LabelEncoder()
        le = le.fit(df_combined[feature])
        df_train[feature] = le.transform(df_train[feature])
        df_test[feature] = le.transform(df_test[feature])
    return df_train, df_test

train_df, test_df = encode_features(train_df, test_df)
train_df.head()
X = train_df.drop(['Survived'], axis=1)
y = train_df['Survived']
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.cross_validation import KFold




#X = train_df.drop(['Survived'], axis=1)
#y = train_df(['Survived'])
clf = RandomForestClassifier()
acc_scorer = make_scorer(accuracy_score)
parameters = {'n_estimators': [4, 6, 9], 
              'max_features': ['log2', 'sqrt','auto'], 
              'criterion': ['entropy', 'gini'],
              'max_depth': [2, 3, 5, 10], 
              'min_samples_split': [2, 3, 5],
              'min_samples_leaf': [1,5,8]
             }

rand = GridSearchCV(clf, parameters, scoring=acc_scorer)
rand = rand.fit(X,y)
clf = rand.best_estimator_
clf.fit(X,y)
predictions = clf.predict(X)
print(accuracy_score(y, predictions))
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()
logreg_acc_scorer = make_scorer(accuracy_score)
logreg_parameters = {'penalty':['l1','l2'],
                     'C':[0.01, 0.1, 1, 10, 100]
                    }
logreg_cv = GridSearchCV(logreg, logreg_parameters, scoring=logreg_acc_scorer)
logreg_cv = logreg_cv.fit(X,y)
logreg_est = logreg_cv.best_estimator_
logreg_est.fit(X,y)
logreg_predictions = logreg_est.predict(X)
print(accuracy_score(y, logreg_predictions))

from sklearn.neural_network import MLPClassifier

nn = MLPClassifier(solver='lbfgs', random_state=1)
nn_acc_scorer = make_scorer(accuracy_score)
nn_params = { 'alpha': [1e-5,0.001,0.01,0.1,1,10,100],
             'hidden_layer_sizes': [(2,2), (2,5), (2,8),(2,10), (3,5), (3,8), (3,10)],
#             'solver': ['lbfgs', 'sgd', 'adam'],
             'solver': ['lbfgs'],
             'random_state': [1],
             'max_iter': [10000]
#             'warm_start': [True, False],
#             'learning_rate': ['constant', 'adaptive']
}
nn_cv = GridSearchCV(nn, nn_params, scoring=nn_acc_scorer)
nn_cv = nn_cv.fit(X,y)
nn_est = nn_cv.best_estimator_
nn_est.fit(X,y)
nn_pred = nn_est.predict(X)
print(accuracy_score(y,nn_pred))
from sklearn.svm import SVC


svc = SVC()
svc_acc_scorer = make_scorer(accuracy_score)
svc_params = { 'C': [0.1, 0.5, 1],
              'kernel': ['poly', 'rbf', 'sigmoid'],
              'degree': [2, 3, 4],
              'decision_function_shape': ['ovo', 'ovr'],
              'class_weight': ['balanced', None]
}
svc_cv = GridSearchCV(svc, svc_params, scoring=svc_acc_scorer)
svc_cv.fit(X,y)
svc_est = svc_cv.best_estimator_
svc_est.fit(X,y)
svc_pred = svc_est.predict(X)
print(accuracy_score(y,svc_pred))
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier()
knn_acc_scorer = make_scorer(accuracy_score)
knn_params = { 'n_neighbors': [2, 4, 8, 16],
              'p': [1, 2, 3],
              'weights': ['uniform', 'distance'],
              'algorithm': [ 'ball_tree', 'kd_tree', 'brute'],
              'n_jobs': [-1]
}

knn_cv = GridSearchCV(knn, knn_params, scoring=knn_acc_scorer)
knn_cv.fit(X,y)
knn_est = knn_cv.best_estimator_
knn_est.fit(X,y)
knn_pred = knn_est.predict(X)
print(accuracy_score(y,knn_pred))
from numpy import loadtxt
from xgboost import XGBClassifier

xgb = XGBClassifier()
print(xgb.get_params().keys())
xgb_acc_scorer = make_scorer(accuracy_score)
xgb_params = {#'Eta': [0.01, 0.015, 0.025, 0.05, 0.1],
              'gamma' : [0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0],
              'max_depth': [3, 5, 7, 9, 12, 15, 17, 25],
              'min_child_weight': [1, 3, 4, 5],
              'subsample' : [0.6, 0.7, 0.8, 0.9, 1],
              'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
              'reg_lambda': [0.01, 0.1, 1.0],
              'reg_alpha': [0, 0.1, 0.5, 1.0]  
}
xgb_cv = GridSearchCV(xgb, xgb_params, scoring=xgb_acc_scorer)
xgb_cv.fit(X,y)
xgb_est = xgb_cv.best_estimator_
xgb_est.fit(X,y)
xgb_pred = xgb_est.predict(X)
print (accuracy_score(y,xgb_pred))