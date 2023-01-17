import numpy as np

import pandas as pd

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn import metrics

from sklearn import preprocessing

from sklearn import svm

from sklearn.model_selection import GridSearchCV

from sklearn import metrics

import tensorflow as tf

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from sklearn.gaussian_process import GaussianProcessClassifier

from sklearn.gaussian_process.kernels import RBF

from sklearn.ensemble import RandomForestClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.naive_bayes import ComplementNB

from sklearn.covariance import EllipticEnvelope

from collections import Counter

from pandas.plotting import scatter_matrix

from sklearn.decomposition import PCA

from sklearn.ensemble import AdaBoostClassifier

from sklearn.tree import DecisionTreeClassifier

import lightgbm as lgb
train_orig = pd.read_csv("../input/train.csv")

test_orig = pd.read_csv("../input/test.csv")
train_orig.info()
train_orig.columns
train_orig.head()
train_orig.describe()


targets = train_orig['target'].value_counts()

ax = targets.plot(kind='bar')

ax.grid('on')

ax.tick_params(rotation=0)
X = train_orig.iloc[:, 2:]

print('Features shape: ', X.shape)

bp_ax = X.boxplot(figsize=(25, 8))

bp_ax.xaxis.label.set_visible(False)
correlations = X.corr()
plt.figure(figsize=(25, 25))

plt.matshow(correlations, fignum=1)

plt.colorbar()
max_corr = -1

min_corr = 1

for i in range(199):

    for j in range(i + 1, 200):

        current = correlations.iloc[i, j]

        min_corr = min(min_corr, current)

        max_corr = max_corr if max_corr > current else current

print('Minimum correlation: ', min_corr)

print('Maximum correlation: ', max_corr)
y = train_orig['target']

print('Targets shape: ', y.shape)
X_test = test_orig.iloc[:, 1:]

print(X_test.shape)
scaler = preprocessing.StandardScaler()

scaler.fit(X)

X = scaler.transform(X)

X_test = scaler.transform(X_test)
seed = 125

X_train, X_dev, y_train, y_dev = train_test_split(X, y, test_size=0.05, random_state=seed, stratify = y)

print('X_train shape: ', X_train.shape)

print('y_train shape: ', y_train.shape)

print('X_dev shape: ', X_dev.shape)

print('y_dev shape: ', y_dev.shape)
models = {}
lr = LogisticRegression(random_state=0, solver='saga', C=0.1).fit(X_train, y_train)

models['LogisticRegression'] = lr
qd = QuadraticDiscriminantAnalysis(reg_param=0.5, tol=1e-10).fit(X_train, y_train)

models['QuadraticDiscriminantAnalysis'] = qd
rfc = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0).fit(X_train, y_train)

models['RandomForestClassifier'] = qd
gclf = GaussianNB().fit(X_train, y_train)

models['GaussianNB'] = qd
for name, model in models.items():

    y_train_predicted = model.predict_proba(X_train)

    train_score = metrics.roc_auc_score(y_train, y_train_predicted[:, 1])

    y_dev_predicted = gclf.predict_proba(X_dev)

    dev_score = metrics.roc_auc_score(y_dev, y_dev_predicted[:, 1])

    print(name, ": ")

    print("Train score: {train}, test score: {test}".format(train=train_score, test=dev_score))
lgb_train = lgb.Dataset(X_train, y_train)

lgb_eval = lgb.Dataset(X_dev, y_dev)



param = {

#     'bagging_freq': 5,

#     'bagging_fraction': 0.4,

    'boost_from_average':'false',

    'boost': 'goss',

    'feature_fraction': 0.1,

    'learning_rate': 0.01,

    'max_depth': -1,

    'metric':'auc',

    'min_data_in_leaf': 100,

    'min_sum_hessian_in_leaf': 10.0,

    'num_leaves': 20,

    'num_threads': 8,

    'tree_learner': 'serial',

    'objective': 'binary',

    'verbosity': 1

}



num_round = 150000

clf = lgb.train(param, lgb_train, num_round, valid_sets = lgb_eval, verbose_eval=100,

                early_stopping_rounds = 3000)
clf.best_iteration
y_train_predicted = clf.predict(X_train, num_iteration=clf.best_iteration)

train_score = metrics.roc_auc_score(y_train, y_train_predicted)

y_dev_predicted = clf.predict(X_dev, num_iteration=clf.best_iteration)

dev_score = metrics.roc_auc_score(y_dev, y_dev_predicted)

print("Train score: {train}, test score: {test}".format(train=train_score, test=dev_score))

print("Confsuion matrix(for treshold 0.5): ")

print("Train set: ")

print(metrics.confusion_matrix(y_train, [1 if x > 0.5 else 0 for x in y_train_predicted]))

print("Dev set: ")

print(metrics.confusion_matrix(y_dev, [1 if x > 0.5 else 0 for x in y_dev_predicted]))
y_test_predicted = clf.predict(X_test, num_iteration=clf.best_iteration)
submission = pd.DataFrame({

        "ID_code": test_orig["ID_code"],

        "target": y_test_predicted

    })

submission.to_csv('sample_submission.csv', index=False)