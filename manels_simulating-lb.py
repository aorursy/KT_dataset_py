import pandas as pd
import numpy as np
import csv
from time import time

import matplotlib.pyplot as plt

import xgboost as xgb

from sklearn.cross_validation import StratifiedKFold
from sklearn.cross_validation import train_test_split

from sklearn.metrics import roc_auc_score
print('Load data...')
train = pd.read_csv("../input/train.csv")
target = train['TARGET']

# Selected features (using forward selection with 4-Fold cv)
selected_features = ['var15', 'saldo_var30', 'num_var22_ult3', 'imp_op_var39_ult1',
                     'num_var45_hace3', 'saldo_medio_var5_hace2', 'var3',
                     'saldo_medio_var8_ult3', 'ind_var41_0']
train = train[selected_features]
print('We consider',len(selected_features),'features')
train, test, target, target_test = train_test_split(train, target, test_size=0.5, random_state=42)
test_Public, test_Private, target_test_Public, target_test_Private = train_test_split(test, target_test, test_size=0.5, random_state=42)
print(train.shape)
print(test.shape)
num_rounds = 100
early_stopping = 200
params = {}
params["objective"] = "binary:logistic"
params["eta"] = 0.07
params["subsample"] = 0.8
params["colsample_bytree"] = 0.8
params["silent"] = 1
params["max_depth"] = 5
params["min_child_weight"] = 1
params["eval_metric"] = "auc"

train0 = np.array(train)
test0 = np.array(test)
test_Public0 = np.array(test_Public)
test_Private0 = np.array(test_Private)

X_train = train0
y_train = target.values

n_seed = 10
train_pred = np.ones(train.shape[0])
train_predictions = pd.DataFrame({"pred1": np.zeros(train.shape[0])})
test_predictions = pd.DataFrame({"pred1": np.zeros(test.shape[0])})
test_predictions_Public = pd.DataFrame({"pred1": np.zeros(test_Public.shape[0])})
test_predictions_Private = pd.DataFrame({"pred1": np.zeros(test_Private.shape[0])})

scores_train = []
scores_test = []
scores_test_Public = []
scores_test_Private = []
# The folds are made by preserving the percentage of samples for each class!
cv_n = 4
kf = StratifiedKFold(target.values, n_folds=cv_n, shuffle=True)

for i in range(0,n_seed):
    params['seed'] = 1+i
    print('Making prediction for seed',params['seed'],'...')
    for cv_train_index, cv_test_index in kf:
        X_train, X_test = train0[cv_train_index, :], train0[cv_test_index, :]
        y_train, y_test = target.iloc[cv_train_index].values, target.iloc[cv_test_index].values
      # train machine learning
        xg_train = xgb.DMatrix(X_train, label=y_train)
        xg_test_cv = xgb.DMatrix(X_test, label=y_test)

        watchlist = [(xg_train, 'train'), (xg_test_cv, 'test')]

        xgclassifier_cv = xgb.train(params, xg_train, 
                                    num_rounds, watchlist,
                                    early_stopping_rounds=early_stopping,
                                    verbose_eval = False);

        # predict
        train_pred[cv_test_index] = xgclassifier_cv.predict(xg_test_cv)

    # Prediction for the training set
    train_predictions['pred'+str(params['seed'])] = train_pred
    scores_train.append(roc_auc_score(target, train_pred))
    print('AUC for train',roc_auc_score(target, train_pred))
    # train machine learning
    xg_train = xgb.DMatrix(X_train, label=y_train)
    xg_test = xgb.DMatrix(test0)
    xg_test_Public = xgb.DMatrix(test_Public0)
    xg_test_Private = xgb.DMatrix(test_Private0)

    watchlist = [(xg_train, 'train')]
    xgclassifier = xgb.train(params, xg_train, num_rounds, watchlist, verbose_eval=False);

    #Predicting test
    test_pred = xgclassifier.predict(xg_test)
    test_predictions['pred'+str(params['seed'])] = test_pred
    #Predicting test public
    test_pred_Public = xgclassifier.predict(xg_test_Public)
    test_predictions_Public['pred'+str(params['seed'])] = test_pred_Public
    #Predicting test private
    test_pred_Private = xgclassifier.predict(xg_test_Private)
    test_predictions_Private['pred'+str(params['seed'])] = test_pred_Private
    
    scores_test.append(roc_auc_score(target_test, test_pred))
    scores_test_Public.append(roc_auc_score(target_test_Public, test_pred_Public))
    scores_test_Private.append(roc_auc_score(target_test_Private, test_pred_Private))
    print('AUC for test',roc_auc_score(target_test, test_pred))
    print('AUC for test Public',roc_auc_score(target_test_Public, test_pred_Public))
    print('AUC for test Private',roc_auc_score(target_test_Private, test_pred_Private))
plt.figure(figsize=(15,10))

#Train-Test
xlim0 = np.min(scores_train)-0.001
xlim1 = np.max(scores_train)+0.001
ylim0 = np.min(scores_test)-0.001
ylim1 = np.max(scores_test)+0.001

blend_train = train_predictions.mean(axis=1)
auc_blend_train = roc_auc_score(target, blend_train)
blend_test = test_predictions.mean(axis=1)
auc_blend_test = roc_auc_score(target_test, blend_test)
# plot with various axes scales
plt.subplot(221)
plt.plot(scores_train, scores_test,'ro')
plt.plot(auc_blend_train,auc_blend_test,'bo')
plt.title('AUC train vs test')
plt.axis([xlim0,xlim1,ylim0,ylim1])
plt.xlabel('train')
plt.ylabel('test')

#Train-Test
xlim0 = np.min(scores_test_Public)-0.001
xlim1 = np.max(scores_test_Public)+0.001
ylim0 = np.min(scores_test_Private)-0.001
ylim1 = np.max(scores_test_Private)+0.001

blend_test_Public = test_predictions_Public.mean(axis=1)
auc_blend_test_Public = roc_auc_score(target_test_Public, blend_test_Public)
blend_test_Private = test_predictions_Private.mean(axis=1)
auc_blend_test_Private = roc_auc_score(target_test_Private, blend_test_Private)
# plot with various axes scales
plt.subplot(222)
plt.plot(scores_test_Public, scores_test_Private,'ro')
plt.plot(auc_blend_test_Public,auc_blend_test_Private,'bo')
plt.title('AUC Public vs Private')
plt.axis([xlim0,xlim1,ylim0,ylim1])
plt.xlabel('Public')
plt.ylabel('Private')

plt.show()