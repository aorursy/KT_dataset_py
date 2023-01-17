import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns



import os

print(os.listdir("../input"))
data_train = pd.read_csv('../input/train.csv')
data_train.head()
data_train.shape
data_train.isnull().sum().any()
data_train['target'].value_counts()
X = data_train.drop(['ID_code', 'target'], axis=1)

y = data_train['target'].values
from sklearn.decomposition import PCA

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import RobustScaler

from sklearn.model_selection import StratifiedShuffleSplit, train_test_split

from sklearn.model_selection import cross_val_score

from sklearn.linear_model import LogisticRegressionCV, LogisticRegression

from sklearn.ensemble import RandomForestClassifier
scaler = RobustScaler()

X_scaled = scaler.fit_transform(X)

pca = PCA(0.99, random_state=1)

pca.fit(X_scaled)
pca.explained_variance_ratio_[:198].sum()
pca.n_components_
X_train, X_test, y_train, y_test = train_test_split(X_scaled, 

                                                    y, 

                                                    test_size=0.1, 

                                                    stratify=y, 

                                                    random_state=1)
kfolds = StratifiedShuffleSplit(n_splits=5, random_state=1)
model_lr = LogisticRegressionCV(Cs=10, cv=kfolds, random_state=1, class_weight='balanced', scoring='roc_auc')
model_lr.fit(X_train, y_train)
model_lr.score(X_test, y_test)
model_lr.C_
from tqdm import tqdm

from sklearn.metrics import roc_auc_score



#model_rf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=1)

model_lr = LogisticRegression(C=0.0001, solver='lbfgs', random_state=1)

model = model_lr

scores = []

scores_xtest = []

preds_xtest = np.empty((X_test.shape[0], 5))

for i, (id_train, id_val) in tqdm(enumerate(kfolds.split(X_train, y_train))):

    model.fit(X_train[id_train], y_train[id_train])

    pred_i = model.predict_proba(X_train[id_val])[:, 1]

    score_i = roc_auc_score(y_train[id_val], pred_i)

    scores.append(score_i)

    pred_xtest = model.predict_proba(X_test)[:, 1]

    preds_xtest[:, i] = pred_xtest

    score_xtest = roc_auc_score(y_test, pred_xtest)

    scores_xtest.append(score_xtest)
print(scores)
print("Mean validation auc: {:.4f} +/- {:.4f}".format(np.mean(scores), np.std(scores)))
print("Mean test auc: {:.6f} +/- {:.6f}".format(np.mean(scores_xtest), np.std(scores_xtest))) 
mean_preds_test = preds_xtest.mean(axis=1)
roc_auc_score(y_test, mean_preds_test)
model.fit(X_train, y_train)
full_preds_test = model.predict_proba(X_test)[:, 1]

roc_auc_score(y_test, full_preds_test)
cv_scores = cross_val_score(model_lr, X_train, y_train, scoring='roc_auc', cv=kfolds, n_jobs=-1)
cv_scores
print("Mean validation auc: {:.4f} +/- {:.4f}".format(np.mean(cv_scores), np.std(cv_scores)))
from sklearn.model_selection import GridSearchCV



model = LogisticRegression(class_weight='balanced', solver='lbfgs', random_state=1)

param_grid = {'C': np.logspace(-4, 4, base=10, num=4)}

grid = GridSearchCV(model, cv=kfolds, param_grid=param_grid, n_jobs=-1, scoring='roc_auc')
grid.fit(X_train, y_train)
grid.best_params_
grid.best_score_
cv_results = list(zip(grid.cv_results_['mean_test_score'], 

                      grid.cv_results_['std_test_score']))



for i, param in enumerate(grid.cv_results_['params']):

    print("C: {:.4f} => mean validation auc: {:.4f} +/- {:.4f}"

          .format(param['C'], cv_results[i][0], cv_results[i][1]))
grid.best_estimator_
grid.best_estimator_.score(X_test, y_test)
grid.score(X_test, y_test)
(grid.best_estimator_.predict(X_test) != grid.predict(X_test)).sum()
from sklearn.model_selection import RandomizedSearchCV



model = LogisticRegression(class_weight='balanced', solver='lbfgs', random_state=1)

param_grid = {'C': np.logspace(-4, 4, base=10, num=100)}

grid_random = RandomizedSearchCV(model, cv=kfolds, n_iter=4, 

                                 param_distributions=param_grid, n_jobs=-1, 

                                 random_state=1, scoring='roc_auc')
grid_random.fit(X_train, y_train)
grid_random.best_params_
grid_random.best_score_
grid_random.cv_results_['params']
cv_results = list(zip(grid_random.cv_results_['mean_test_score'], 

                      grid_random.cv_results_['std_test_score']))



for i, param in enumerate(grid_random.cv_results_['params']):

    print("C: {:.4f} => mean validation auc: {:.4f} +/- {:.4f}"

          .format(param['C'], cv_results[i][0], cv_results[i][1]))