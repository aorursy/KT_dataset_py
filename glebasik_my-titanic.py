from sklearn import tree

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV

from sklearn.metrics import precision_score, recall_score, f1_score, roc_curve, auc



import xgboost as xgb



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



import warnings

warnings.filterwarnings('ignore')
sns.set(rc={'figure.figsize': (9, 6)})
titanic_data = pd.read_csv('../input/titanic/train.csv')
titanic_data.head()
titanic_data.isnull().sum()
X = titanic_data.drop(columns=['Name', 'Ticket', 'Cabin'])
fem = X.query('Sex == "female"')

fem = fem.fillna({'Age': fem.Age.median()})

ml = X.query('Sex == "male"')

ml = ml.fillna({'Age': ml.Age.median()})

X = fem.append(ml)

y = X.Survived

X = X.drop(columns=['Survived', 'PassengerId'])

X = pd.get_dummies(X)

X = X.fillna({'Fare': X.Fare.median()}).drop(columns=['Sex_male'])
X.tail()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
clf = tree.DecisionTreeClassifier()
parameters = {'criterion': ['gini', 'entropy'], 'max_depth': range(3, 15),

              'min_samples_split': range(50, 500, 10),

              'min_samples_leaf': range(50, 500, 10)}
rand_search_cv_clf = RandomizedSearchCV(clf, parameters, cv=5, scoring='roc_auc',

                                        n_iter=500, random_state=42, n_jobs=-1)
rand_search_cv_clf.fit(X_train, y_train)
rand_search_cv_clf.best_params_
tree_best_clf = rand_search_cv_clf.best_estimator_
tree_best = np.array([tree_best_clf.score(X_test, y_test), precision_score(y_test,

                                                                           tree_best_clf.predict(X_test)),

      recall_score(y_test, tree_best_clf.predict(X_test)),

      f1_score(y_test, tree_best_clf.predict(X_test))])

tree_best
y_predicted_prob = tree_best_clf.predict_proba(X_test)

tree_fpr, tree_tpr, thresholds = roc_curve(y_test, y_predicted_prob[:,1])

tree_roc_auc= auc(tree_fpr, tree_tpr)
clf = LogisticRegression()
parameters = {'solver': ['newton-cg', 'lbfgs', 'sag', 'saga'], 

              'max_iter': range(2000, 5000, 100), 

              'penalty': ['none', 'l2']}
rand_search_cv_clf = RandomizedSearchCV(clf, parameters, cv=5, scoring='roc_auc',

                                        n_iter=30, random_state=42, n_jobs=-1)
rand_search_cv_clf.fit(X_train, y_train)
rand_search_cv_clf.best_params_
regr_best_clf = rand_search_cv_clf.best_estimator_
regr_best = np.array([regr_best_clf.score(X_test, y_test), precision_score(y_test,

                                                                           regr_best_clf.predict(X_test)),

      recall_score(y_test, regr_best_clf.predict(X_test)),

      f1_score(y_test, regr_best_clf.predict(X_test))])

regr_best
y_predicted_prob = regr_best_clf.predict_proba(X_test)

regr_fpr, regr_tpr, thresholds = roc_curve(y_test, y_predicted_prob[:,1])

regr_roc_auc= auc(regr_fpr, regr_tpr)
clf = RandomForestClassifier()
parameters = {'n_estimators': range(10, 150, 5), 'criterion': ['gini', 'entropy'],

              'max_depth': range(3, 15),

              'min_samples_split': range(10, 500, 10),

              'min_samples_leaf': range(10, 500, 10)}
rand_search_cv_clf = RandomizedSearchCV(clf, parameters, cv=5, scoring='roc_auc',

                                        n_iter=300, random_state=42, n_jobs=-1)
rand_search_cv_clf.fit(X_train, y_train)
rand_search_cv_clf.best_params_
forest_best_clf = rand_search_cv_clf.best_estimator_
forest_best = np.array([forest_best_clf.score(X_test, y_test), precision_score(y_test,

                                                                               forest_best_clf.predict(X_test)),

      recall_score(y_test, forest_best_clf.predict(X_test)),

      f1_score(y_test, forest_best_clf.predict(X_test))])

forest_best
y_predicted_prob = forest_best_clf.predict_proba(X_test)

forest_fpr, forest_tpr, thresholds = roc_curve(y_test, y_predicted_prob[:,1])

forest_roc_auc= auc(forest_fpr, forest_tpr)
clf = xgb.XGBClassifier()
parameters = {'learning_rate': [0.1, 0.3, 0.6, 0.9],

              'verbosity': range(0, 4),

              'booster': ['gbtree', 'gblinear', 'dart'],

              'max_depth': range(3, 15), 'min_samples_split': range(10, 1000, 10),

              'min_samples_leaf': range(10, 1000, 10),

              'n_estimators': range(10, 150, 5)}
rand_search_cv_clf = RandomizedSearchCV(clf, parameters, cv=5, scoring='roc_auc',

                                        n_iter=300, random_state=42, n_jobs=-1)
rand_search_cv_clf.fit(X_train, y_train)
rand_search_cv_clf.best_params_
xgb_best_clf = rand_search_cv_clf.best_estimator_
xgb_best = np.array([xgb_best_clf.score(X_test, y_test), precision_score(y_test,

                                                                               xgb_best_clf.predict(X_test)),

      recall_score(y_test, xgb_best_clf.predict(X_test)),

      f1_score(y_test, xgb_best_clf.predict(X_test))])

xgb_best
y_predicted_prob = xgb_best_clf.predict_proba(X_test)

xgb_fpr, xgb_tpr, thresholds = roc_curve(y_test, y_predicted_prob[:,1])

xgb_roc_auc= auc(xgb_fpr, xgb_tpr)
clf = xgb.XGBRFClassifier()
parameters = {'learning_rate': [0.1, 0.3, 0.5, 1, 3, 5, 10],

              'verbosity': range(0, 4),

              'booster': ['gbtree', 'gblinear', 'dart'],

              'max_depth': range(3, 15), 'min_samples_split': range(10, 1000, 10),

              'min_samples_leaf': range(10, 1000, 10),

              'n_estimators': range(10, 150, 5)}
rand_search_cv_clf = RandomizedSearchCV(clf, parameters, cv=5, scoring='roc_auc',

                                        n_iter=300, random_state=42, n_jobs=-1)
rand_search_cv_clf.fit(X_train, y_train)
rand_search_cv_clf.best_params_
xgbrf_best_clf = rand_search_cv_clf.best_estimator_
xgbrf_best = np.array([xgbrf_best_clf.score(X_test, y_test), precision_score(y_test,

                                                                               xgbrf_best_clf.predict(X_test)),

      recall_score(y_test, xgbrf_best_clf.predict(X_test)),

      f1_score(y_test, xgbrf_best_clf.predict(X_test))])

xgbrf_best
y_predicted_prob = xgbrf_best_clf.predict_proba(X_test)

xgbrf_fpr, xgbrf_tpr, thresholds = roc_curve(y_test, y_predicted_prob[:,1])

xgbrf_roc_auc= auc(xgbrf_fpr, xgbrf_tpr)
scores = pd.DataFrame([regr_best.T, tree_best.T, forest_best.T, xgb_best.T, xgbrf_best.T], 

                      index=['logistic_regression', 'decision_tree', 'random_forest', 'XGB', 'XGBRF'],

                      columns=['score', 'precision', 'recall', 'f1'])
scores
plt.figure()

plt.plot(regr_fpr, regr_tpr,

          label='Logistic regression ROC curve (area = %0.3f)' % regr_roc_auc)

plt.plot(tree_fpr, tree_tpr,

          label='Decision tree ROC curve (area = %0.3f)' % tree_roc_auc)

plt.plot(forest_fpr, forest_tpr,

          label='Random forest ROC curve (area = %0.3f)' % forest_roc_auc)

plt.plot(xgb_fpr, xgb_tpr,

          label='XGB ROC curve (area = %0.3f)' % xgb_roc_auc)

plt.plot(xgbrf_fpr, xgbrf_tpr,

          label='XGBRF ROC curve (area = %0.3f)' % xgbrf_roc_auc)

plt.plot([0, 1], [0, 1], color='navy',  linestyle='--')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('Receiver operating characteristic example')

plt.legend(loc="lower right")

plt.show()
titanic_test = pd.read_csv('../input/titanic/test.csv') 
X_pred = titanic_test.drop(columns=['Name', 'Ticket', 'Cabin'])
fem = X_pred.query('Sex == "female"')

fem = fem.fillna({'Age': fem.Age.median()})

ml = X_pred.query('Sex == "male"')

ml = ml.fillna({'Age': ml.Age.median()})

X_pred = fem.append(ml)
pasid = X_pred.PassengerId

X_pred = X_pred.drop(columns=['PassengerId'])

X_pred = pd.get_dummies(X_pred)

X_pred = X_pred.fillna({'Fare': X_pred.Fare.median()}).drop(columns=['Sex_male'])
X_pred.head()
X_pred.isnull().sum()
y_pred = xgb_best_clf.predict(X_pred)
pasid = pasid.to_numpy()
y_pred.shape
out = np.hstack((pasid.reshape((418, 1)), y_pred.reshape((418, 1))))
out.shape
out = pd.DataFrame(out, columns=['PassengerId', 'Survived'])
out.to_csv('out_1.csv', index=False)