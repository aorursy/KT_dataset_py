import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import xgboost as xgb

import sklearn

import os

print(os.listdir("../input"))
for i in [np,pd,sns,xgb, sklearn]:

    print(i.__version__)
import subprocess

from IPython.display import Image

from collections import Counter

from sklearn.datasets import make_classification

from sklearn.model_selection import train_test_split

from sklearn.metrics import log_loss, accuracy_score



from sklearn.tree import DecisionTreeClassifier, export_graphviz

from sklearn.ensemble import AdaBoostClassifier

from sklearn.ensemble import GradientBoostingClassifier



seed = 104
X,y= make_classification(n_samples=1000, n_features=20, n_informative=8, n_redundant=3, n_repeated=2, random_state=seed)



X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=seed)
pd.DataFrame(X_train).head(3)
pd.Series(y_train).head(3)
print("Train label distribution:")

print(Counter(y_train))



print("\nTest label distribution:")

print(Counter(y_test))
decision_tree = DecisionTreeClassifier(random_state=seed).fit(X_train, y_train)                # TRAINING THE CLASSIFIER MODEL



decision_tree_y_pred = decision_tree.predict(X_test)                                           # PREDICT THE OUTPUT

decision_tree_y_pred_prob = decision_tree.predict_proba(X_test)



decision_tree_accuracy = accuracy_score(y_test, decision_tree_y_pred)

decision_tree_logloss = log_loss(y_test, decision_tree_y_pred_prob)



print("== Decision Tree ==")

print("Accuracy: {0:.2f}".format(decision_tree_accuracy))

print("Log loss: {0:.2f}".format(decision_tree_logloss))

print("Number of nodes created: {}".format(decision_tree.tree_.node_count))


print('True labels:')

print(y_test[:35,])

print('\nPredicted labels:')

print(decision_tree_y_pred[:35,])

print('\nPredicted probabilities:')

print(decision_tree_y_pred_prob[:5,])

adaboost = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=1), algorithm='SAMME', n_estimators=1000, random_state=seed)



adaboost.fit(X_train, y_train)



adaboost_y_pred = adaboost.predict(X_test)

adaboost_y_pred_proba = adaboost.predict_proba(X_test)



adaboost_accuracy = accuracy_score(y_test, adaboost_y_pred)

adaboost_logloss = log_loss(y_test, adaboost_y_pred_proba)



print("== AdaBoost ==")

print("Accuracy: {0:.2f}".format(adaboost_accuracy))

print("Log loss: {0:.2f}".format(adaboost_logloss))
print('True labels:')

print(y_test[:25,])

print('\nPredicted labels:')

print(adaboost_y_pred[:25,])

print('\nPredicted probabilities:')

print(adaboost_y_pred_proba[:5,])
print("Error: {0:.2f}".format(adaboost.estimator_errors_[0]))

print("Tree importance: {0:.2f}".format(adaboost.estimator_weights_[0]))
gbc = GradientBoostingClassifier(max_depth=1, n_estimators=1000, warm_start=True, random_state=seed)



gbc.fit(X_train, y_train)



gbc_y_pred = gbc.predict(X_test)

gbc_y_pred_proba = gbc.predict_proba(X_test)



gbc_accuracy = accuracy_score(y_test, gbc_y_pred)

gbc_logloss = log_loss(y_test, gbc_y_pred_proba)



print("== Gradient Boosting ==")

print("Accuracy: {0:.2f}".format(gbc_accuracy))

print("Log loss: {0:.2f}".format(gbc_logloss))
print('True labels:')

print(y_test[:25,])

print('\nPredicted labels:')

print(gbc_y_pred[:25,])

print('\nPredicted probabilities:')

print(gbc_y_pred_proba[:5,])
dtrain = xgb.DMatrix('../input/agaricus_train.txt')

dtest = xgb.DMatrix('../input/agaricus_test.txt')
print([w for w in dir(xgb) if not w.startswith( "_")])
print("Train dataset contains {0} rows and {1} columns".format(dtrain.num_row(), dtrain.num_col()))

print("Test dataset contains {0} rows and {1} columns".format(dtest.num_row(), dtest.num_col()))
print("Train possible labels: ", np.unique(dtrain.get_label()))

print("\nTest possible labels: ", np.unique(dtest.get_label()))
params = {

    'objective':'binary:logistic',

    'max_depth':2,

    'silent':1,

    'eta':1

}



num_rounds = 5
bst = xgb.train(params=params, dtrain=dtrain, num_boost_round=num_rounds)
watchlist = [(dtest,'test'), (dtrain,'train')]

bst = xgb.train(params, dtrain, num_rounds, watchlist)
preds_prob = bst.predict(dtest)

preds_prob
labels = dtest.get_label()

preds = preds_prob > 0.5 # threshold

correct = 0



for i in range(len(preds)):

    if (labels[i] == preds[i]):

        correct += 1



print('Predicted correctly: {0}/{1}'.format(correct, len(preds)))

print('Error: {0:.4f}'.format(1-correct/len(preds)))
dtrain = xgb.DMatrix('../input/agaricus_train.txt')

dtest = xgb.DMatrix('../input/agaricus_test.txt')
params = {

    'objective':'binary:logistic',

    'max_depth':1,

    'silent':1,

    'eta':0.5

}



num_rounds = 5
watchlist  = [(dtest,'test'), (dtrain,'train')] # native interface only

bst = xgb.train(params, dtrain, num_rounds, watchlist)
trees_dump = bst.get_dump(fmap='../input/featmap.txt', with_stats=True)



for tree in trees_dump:

    print(tree)
xgb.plot_importance(bst, importance_type='gain', xlabel='Gain')
xgb.plot_importance(bst)
importances = bst.get_fscore()

importances
# create df

importance_df = pd.DataFrame({'Splits': list(importances.values()),'Feature': list(importances.keys())})

importance_df.sort_values(by='Splits', inplace=True)

importance_df.plot(kind='barh', x='Feature', figsize=(8,6), color='orange')
from sklearn.model_selection import validation_curve

from sklearn.datasets import load_svmlight_file

from sklearn.model_selection import StratifiedKFold

from sklearn.datasets import make_classification

from xgboost.sklearn import XGBClassifier

from scipy.sparse import vstack



seed = 123

np.random.seed(seed)
print([w for w in dir(xgb.sklearn) if not w.startswith('_')])
#X, y = make_classification(n_samples=1000, n_features=20, n_informative=8, n_redundant=3, n_repeated=2, random_state=seed)

seed = 2024

X, y = make_classification(n_samples=10000, n_features=30, n_informative=10, n_redundant=5, n_repeated=3, random_state=seed)
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)

cv = skf.split(X,y)

cv 
default_params = {

    'objective': 'binary:logistic',

    'max_depth': 1,

    'learning_rate': 0.3,

    'silent': 1.0

}



n_estimators_range = np.linspace(1, 200, 10).astype('int')



train_scores, test_scores = validation_curve(

    XGBClassifier(**default_params),

    X, y,

    param_name = 'n_estimators',

    param_range = n_estimators_range,

    cv=cv,

    scoring='accuracy'

)
train_scores_mean = np.mean(train_scores, axis=1)

train_scores_std = np.std(train_scores, axis=1)

test_scores_mean = np.mean(test_scores, axis=1)

test_scores_std = np.std(test_scores, axis=1)



fig = plt.figure(figsize=(10, 6), dpi=100)



plt.title("Validation Curve with XGBoost (eta = 0.3)")

plt.xlabel("number of trees")

plt.ylabel("Accuracy")

plt.ylim(0.7, 1.1)



plt.plot(n_estimators_range,

             train_scores_mean,

             label="Training score",

             color="r")



plt.plot(n_estimators_range,

             test_scores_mean, 

             label="Cross-validation score",

             color="g")



plt.fill_between(n_estimators_range, 

                 train_scores_mean - train_scores_std,

                 train_scores_mean + train_scores_std, 

                 alpha=0.2, color="r")



plt.fill_between(n_estimators_range,

                 test_scores_mean - test_scores_std,

                 test_scores_mean + test_scores_std,

                 alpha=0.2, color="g")



plt.axhline(y=1, color='k', ls='dashed')



plt.legend(loc="best")

plt.show()



i = np.argmax(test_scores_mean)

print("Best cross-validation result ({0:.2f}) obtained for {1} trees".format(test_scores_mean[i], n_estimators_range[i]))
default_params = {

    'objective': 'binary:logistic',

    'max_depth': 2, # changed

    'learning_rate': 0.3,

    'silent': 1.0,

    'colsample_bytree': 0.6, # added

    'subsample': 0.7 # added

}



n_estimators_range = np.linspace(1, 200, 10).astype('int')



train_scores, test_scores = validation_curve(

    XGBClassifier(**default_params),

    X, y,

    param_name = 'n_estimators',

    param_range = n_estimators_range,

    cv=cv,

    scoring='accuracy'

)
train_scores_mean = np.mean(train_scores, axis=1)

train_scores_std = np.std(train_scores, axis=1)

test_scores_mean = np.mean(test_scores, axis=1)

test_scores_std = np.std(test_scores, axis=1)



fig = plt.figure(figsize=(10, 6), dpi=100)



plt.title("Validation Curve with XGBoost (eta = 0.3)")

plt.xlabel("number of trees")

plt.ylabel("Accuracy")

plt.ylim(0.7, 1.1)



plt.plot(n_estimators_range,

             train_scores_mean,

             label="Training score",

             color="r")



plt.plot(n_estimators_range,

             test_scores_mean, 

             label="Cross-validation score",

             color="g")



plt.fill_between(n_estimators_range, 

                 train_scores_mean - train_scores_std,

                 train_scores_mean + train_scores_std, 

                 alpha=0.2, color="r")



plt.fill_between(n_estimators_range,

                 test_scores_mean - test_scores_std,

                 test_scores_mean + test_scores_std,

                 alpha=0.2, color="g")



plt.axhline(y=1, color='k', ls='dashed')



plt.legend(loc="best")

plt.show()



i = np.argmax(test_scores_mean)

print("Best cross-validation result ({0:.2f}) obtained for {1} trees".format(test_scores_mean[i], n_estimators_range[i]))
from xgboost.sklearn import XGBClassifier



from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from sklearn.datasets import make_classification

from sklearn.model_selection import StratifiedKFold



from scipy.stats import randint, uniform



# reproducibility

seed = 342

np.random.seed(seed)
seed = 2024

X, y = make_classification(n_samples=1000, n_features=30, n_informative=10,

                           n_redundant=5, n_repeated=3, random_state=seed)

skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)

cv = skf.split(X,y)
print(X.shape)

print(y.shape)

print(X[:2,:])
type(train_index)
params_grid = {

    'max_depth': [1, 2, 3],

    'n_estimators': [5, 10, 25, 50],

    'learning_rate': np.linspace(1e-16, 1, 3)

}
params_fixed = {

    'objective': 'binary:logistic',

    'silent': 1

}
bst_grid = GridSearchCV(estimator=XGBClassifier(**params_fixed, seed=seed),param_grid=params_grid,cv=cv,scoring='accuracy')
bst_grid.fit(X, y)
bst_grid.cv_results_
print("Best accuracy obtained: {0}".format(bst_grid.best_score_))

print("Parameters:")

for key, value in bst_grid.best_params_.items():

    print("\t{}: {}".format(key, value))
params_dist_grid = {

    'max_depth': [1, 2, 3, 4],

    'gamma': [0, 0.5, 1],

    'n_estimators': randint(1, 1001), # uniform discrete random distribution

    'learning_rate': uniform(), # gaussian distribution

    'subsample': uniform(), # gaussian distribution

    'colsample_bytree': uniform() # gaussian distribution

}
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)

cv = skf.split(X,y)



rs_grid = RandomizedSearchCV(

    estimator=XGBClassifier(**params_fixed, seed=seed),

    param_distributions=params_dist_grid,

    n_iter=10,

    cv=cv,

    scoring='accuracy',

    random_state=seed

)
rs_grid.fit(X, y)
print([w for w in dir(rs_grid) if not w.startswith('_')])
print(rs_grid.best_estimator_)

print(rs_grid.best_params_)

print(rs_grid.best_score_)

#print(rs_grid.score)
params = {

    'objective':'binary:logistic',

    'max_depth':1,

    'silent':1,

    'eta':0.5

}



num_rounds = 5

watchlist  = [(dtest,'test'), (dtrain,'train')]



bst = xgb.train(params, dtrain, num_rounds, watchlist)
params['eval_metric'] = 'logloss'

bst = xgb.train(params, dtrain, num_rounds, watchlist)
params['eval_metric'] = ['logloss', 'auc']

bst = xgb.train(params, dtrain, num_rounds, watchlist)
# custom evaluation metric

def misclassified(pred_probs, dtrain):

    labels = dtrain.get_label() # obtain true labels

    preds = pred_probs > 0.5 # obtain predicted values

    return 'misclassified', np.sum(labels != preds)
bst = xgb.train(params, dtrain, num_rounds, watchlist, feval=misclassified, maximize=False)
evals_result = {}

bst = xgb.train(params, dtrain, num_rounds, watchlist, feval=misclassified, maximize=False, evals_result=evals_result)
from pprint import pprint

pprint(evals_result)
params['eval_metric'] = 'error'

num_rounds = 1500



bst = xgb.train(params, dtrain, num_rounds, watchlist, early_stopping_rounds=10)
print("Booster best train score: {}".format(bst.best_score))

print("Booster best iteration: {}".format(bst.best_iteration))

print("Booster best number of trees limit: {}".format(bst.best_ntree_limit))
num_rounds = 10 # how many estimators

hist = xgb.cv(params, dtrain, num_rounds, nfold=10, metrics={'error'}, seed=seed)

hist
# create valid dataset

np.random.seed(seed)



data_v = np.random.rand(10,5) # 10 entities, each contains 5 features

data_v
# add some missing values

data_m = np.copy(data_v)



data_m[2, 3] = np.nan

data_m[0, 1] = np.nan

data_m[0, 2] = np.nan

data_m[1, 0] = np.nan

data_m[4, 4] = np.nan

data_m[7, 2] = np.nan

data_m[9, 1] = np.nan



data_m
np.random.seed(seed)



label = np.random.randint(2, size=10) # binary target

label
# specify general training parameters

params = {

    'objective':'binary:logistic',

    'max_depth':1,

    'silent':1,

    'eta':0.5

}



num_rounds = 5
dtrain_v = xgb.DMatrix(data_v, label=label)



xgb.cv(params, dtrain_v, num_rounds, seed=seed)

dtrain_m = xgb.DMatrix(data_m, label=label, missing=np.nan)



xgb.cv(params, dtrain_m, num_rounds, seed=seed)
params = {

    'objective': 'binary:logistic',

    'max_depth': 1,

    'learning_rate': 0.5,

    'silent': 1.0,

    'n_estimators': 5

}



clf = XGBClassifier(**params)

clf
from sklearn.model_selection import cross_val_score

cross_val_score(clf, data_v, label, cv=2, scoring='accuracy')
cross_val_score(clf, data_m, label, cv=2, scoring='accuracy')
from sklearn.datasets import make_classification

from sklearn.metrics import accuracy_score, precision_score, recall_score

from sklearn.model_selection import train_test_split



# reproducibility

seed = 123
X, y = make_classification(

    n_samples=200,

    n_features=5,

    n_informative=3,

    n_classes=2,

    weights=[.9, .1],

    shuffle=True,

    random_state=seed

)



print('There are {} positive instances.'.format(y.sum()))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, stratify=y, random_state=seed)



print('Total number of postivie train instances: {}'.format(y_train.sum()))

print('Total number of positive test instances: {}'.format(y_test.sum()))
dtrain = xgb.DMatrix(X_train, label=y_train)

dtest = xgb.DMatrix(X_test)
params = {

    'objective':'binary:logistic',

    'max_depth':1,

    'silent':1,

    'eta':1

}



num_rounds = 15
bst = xgb.train(params, dtrain, num_rounds)

y_test_preds = (bst.predict(dtest) > 0.5).astype('int')

pd.crosstab(

    pd.Series(y_test, name='Actual'),

    pd.Series(y_test_preds, name='Predicted'),

    margins=True

)
print('Accuracy: {0:.2f}'.format(accuracy_score(y_test, y_test_preds)))

print('Precision: {0:.2f}'.format(precision_score(y_test, y_test_preds)))

print('Recall: {0:.2f}'.format(recall_score(y_test, y_test_preds)))
weights = np.zeros(len(y_train))

weights[y_train == 0] = 1

weights[y_train == 1] = 5



dtrain = xgb.DMatrix(X_train, label=y_train, weight=weights) # weights added

dtest = xgb.DMatrix(X_test)
bst = xgb.train(params, dtrain, num_rounds)

y_test_preds = (bst.predict(dtest) > 0.5).astype('int')



pd.crosstab(

    pd.Series(y_test, name='Actual'),

    pd.Series(y_test_preds, name='Predicted'),

    margins=True

)
print('Accuracy: {0:.2f}'.format(accuracy_score(y_test, y_test_preds)))

print('Precision: {0:.2f}'.format(precision_score(y_test, y_test_preds)))

print('Recall: {0:.2f}'.format(recall_score(y_test, y_test_preds)))
dtrain = xgb.DMatrix(X_train, label=y_train)

dtest = xgb.DMatrix(X_test)
train_labels = dtrain.get_label()



ratio = float(np.sum(train_labels == 0)) / np.sum(train_labels == 1)

params['scale_pos_weight'] = ratio
bst = xgb.train(params, dtrain, num_rounds)

y_test_preds = (bst.predict(dtest) > 0.5).astype('int')



pd.crosstab(

    pd.Series(y_test, name='Actual'),

    pd.Series(y_test_preds, name='Predicted'),

    margins=True

)
print('Accuracy: {0:.2f}'.format(accuracy_score(y_test, y_test_preds)))

print('Precision: {0:.2f}'.format(precision_score(y_test, y_test_preds)))

print('Recall: {0:.2f}'.format(recall_score(y_test, y_test_preds)))