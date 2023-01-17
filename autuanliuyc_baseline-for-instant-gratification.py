%matplotlib inline

%reload_ext autoreload

%autoreload 2

# 多行输出

from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = "all"
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))

from fastai.tabular import *

from fastai.callbacks import EarlyStoppingCallback

from sklearn.feature_selection import VarianceThreshold

from sklearn.model_selection import StratifiedKFold

from sklearn.preprocessing import StandardScaler

from sklearn import svm, neighbors

from sklearn.metrics import roc_auc_score

import lightgbm as lgb

from sklearn.model_selection import train_test_split

from sklearn.decomposition import PCA

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# Any results you write to the current directory are saved as output.
lda = LinearDiscriminantAnalysis(solver='svd', n_components=40, shrinkage=None)

pca = PCA(n_components=40)
root = Path("../input")

train_df = pd.read_csv(root/'train.csv')

test_df = pd.read_csv(root/'test.csv')

train_df['wheezy-copper-turtle-magic'] = train_df['wheezy-copper-turtle-magic'].astype('category')

test_df['wheezy-copper-turtle-magic'] = test_df['wheezy-copper-turtle-magic'].astype('category')

submission = pd.read_csv(root/'sample_submission.csv')
# train_df.head()
# train_df.describe()
# train_df.info()
# test_df.head()
# test_df.describe()
# test_df.info()
# submission.head()
# y = train_df.groupby('target').count().iloc[:, 0]

# y

# wt = y.values/sum(y.values)

# wt
# procs = [FillMissing, Categorify, Normalize]
# valid_idx = range(round(len(train_df) * 0.9), len(train_df))
y = train_df.groupby('wheezy-copper-turtle-magic').count().iloc[:, :1]
plt.figure(figsize=(20, 16))

plt.bar(y.index, y['id'])
all_cols = test_df.columns

dep_var = 'target'

cat_names = ['wheezy-copper-turtle-magic']

cont_names = list(set(all_cols) - set(['id', 'wheezy-copper-turtle-magic']))
test = TabularList.from_df(test_df, cat_names=cat_names, cont_names=cont_names)
# data = (TabularList.from_df(train_df, cat_names=cat_names, cont_names=cont_names, procs=procs)

#         .split_by_rand_pct(0.1, seed=123)

#         .label_from_df(cols=dep_var)

#         .add_test(test)

#         .databunch(path='.', device=torch.device('cuda: 0'))

#        )
# data.show_batch(rows=5)
# len(data.test_ds.cont_names)
# learn = tabular_learner(data, layers=[512, 128], ps=[0.001, 0.001], metrics=accuracy, emb_drop=0.001,

#                         callback_fns=[partial(EarlyStoppingCallback, monitor='accuracy', min_delta=0.0001, patience=5)]).to_fp16()

# learn = tabular_learner(data, layers=[2000, 1000, 500, 100], ps=[0.3, 0.3, 0.2, 0.2], metrics=accuracy, emb_drop=0.01,

#                         callback_fns=[partial(EarlyStoppingCallback, monitor='accuracy', min_delta=0.0001, patience=5)]).to_fp16()
# learn.model
# learn.opt_func
# learn.loss_func
# learn.lr_find()

# learn.recorder.plot()
# lr = 1e-2
# learn.fit_one_cycle(100, slice(lr))
# learn.recorder.plot_losses()
# preds, _ = learn.get_preds(ds_type=DatasetType.Test)
# submission['target'] = preds[:, 1].numpy()
# submission.head()
# submission.to_csv('submission1.csv', index=None, encoding='utf-8')
preds_ = np.zeros(len(test_df))

preds_train = np.zeros(len(train_df))

procs1 = [FillMissing, Normalize]
for cat in range(512):

    train1 = train_df[(train_df[cat_names]==cat).values.reshape(-1,)]

    test1 = test_df[(test_df[cat_names]==cat).values.reshape(-1,)]

    idx = test1.index

    idx1 = train1.index

        

    # feature selection

    fs = VarianceThreshold(threshold=1.5).fit(train1[cont_names])

    train2 = fs.transform(train1[cont_names])

    test2 = fs.transform(test1[cont_names])

    cols = []

    for i in range(len(cont_names)):

        if fs.variances_[i] > 1.5:

            cols.append(cont_names[i])

    train3 = pd.DataFrame(train2, columns=cols)

    test3 = pd.DataFrame(test2, columns=cols)

    train3[dep_var] = train1[dep_var].values  # keep same index

    

    # reset index

    train3.reset_index(drop=True, inplace=True)

    # Do not reset test set's index

    

    # cv

    folds = 10

    cv = StratifiedKFold(n_splits=folds, random_state=42)

    for train_idx, val_idx in cv.split(train2, train3[dep_var]):

        # make data

        data = (TabularList.from_df(train3.iloc[train_idx, :], cat_names=None, cont_names=cols, procs=procs1)

            .split_by_rand_pct(0.1, seed=123)

            .label_from_df(cols=dep_var)

            .add_test(TabularList.from_df(test3, cat_names=None, cont_names=cols))

            .databunch(path='.', device=torch.device('cuda: 0'))

           )

    

        # model

        learn = tabular_learner(data, layers=[256, 128], ps=[0.00, 0.0], metrics=accuracy, emb_drop=0.001,

                            callback_fns=[partial(EarlyStoppingCallback, monitor='accuracy', min_delta=0.001, patience=3)]).to_fp16()

        learn.fit_one_cycle(100, slice(1e-3))



        # predict

        preds, _ = learn.get_preds(ds_type=DatasetType.Test)

        preds_[idx] += preds.numpy()[:,1]/folds
# submission['target'] = preds_
# submission.to_csv('submission.csv', index=None, encoding='utf-8')
# new_train = train_df[cat_names + cont_names]

# new_test = test_df[cat_names + cont_names]
# new_train.head()
# data, target = new_train.values, train_df['target'].values

# X_train, X_valid, y_train, y_valid = train_test_split(data, target, test_size=0.1, random_state=123)

# X_test = test_df[cat_names + cont_names].values
# params = {

#     'boosting_type': 'gbdt',

#     'objective': 'binary',

#     'metric': ['auc', 'binary_logloss'],

#     'num_leaves': 31,

#     'learning_rate': 0.08,

#     'n_estimators': 2000,

# #     'slient': True,

# #     'reg_alpha': 0.001

# }



# gbm = lgb.LGBMClassifier(**params)

# feature_names = cat_names + cont_names



# # 训练

# gbm.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], 

#         feature_name=feature_names, categorical_feature=cat_names,

#         eval_metric=['binary_logloss', 'auc'], early_stopping_rounds=15, verbose=False)
# ys = gbm.predict_proba(X_valid, num_iteration=gbm.best_iteration_)

# roc_auc_score(y_valid, ys[:, 1])
# predict

# y_pred = gbm.predict(X_valid, num_iteration=gbm.best_iteration_)

# print(accuracy_score(y_valid, y_pred))
# y_pred = gbm.predict_proba(X_test, num_iteration=gbm.best_iteration_)
# submission['target'] = y_pred[:, 1]
# submission.head()
# submission.to_csv('submission.csv', index=None, encoding='utf-8')
# preds_test = np.zeros(len(test_df))

# preds_train = np.zeros(len(train_df))

# params = {

#     'boosting_type': 'gbdt',

#     'objective': 'binary',

#     'metric': ['auc', 'binary_logloss'],

#     'num_leaves': 31,

#     'learning_rate': 0.08,

#     'n_estimators': 2000,

#     'slient': True

# }
# for cat in range(512):

#     train1 = train_df[(train_df[cat_names]==cat).values.reshape(-1,)]

#     test1 = test_df[(test_df[cat_names]==cat).values.reshape(-1,)]

#     idx = test1.index

#     idx1 = train1.index

    

#     # feature selection

#     fs = VarianceThreshold(threshold=1.5).fit(train1[cont_names])

#     train2 = fs.transform(train1[cont_names])

#     test2 = fs.transform(test1[cont_names])

    

    

#     cols = []

#     for i in range(len(cont_names)):

#         if fs.variances_[i] > 1.5:

#             cols.append(cont_names[i])

#     train3 = pd.DataFrame(train2, columns=cols)

#     test3 = pd.DataFrame(test2, columns=cols)

#     train3[dep_var] = train1[dep_var].values  # keep same index

    

#     # reset index for cv

#     train3.reset_index(drop=True, inplace=True)

#     # Do not reset test set's index

    

#     # cv

#     folds, aucs = 10, []

#     cv = StratifiedKFold(n_splits=folds, random_state=42)

#     for train_idx, val_idx in cv.split(train2, train3[dep_var].values):

#         # make data

#         gbm = lgb.LGBMClassifier(**params)

#         _ = gbm.fit(train2[train_idx], train3[dep_var].values[train_idx], eval_set=[(train2[val_idx], train3[dep_var].values[val_idx])], 

#         feature_name=cols, eval_metric=['binary_logloss', 'auc'], early_stopping_rounds=5, verbose=False)

        

#         # predict

#         preds = gbm.predict_proba(test2, num_iteration=gbm.best_iteration_)

#         preds_test[idx] += preds[:,1]/folds

        

#         # eval

#         preds_train[idx1[val_idx]] = gbm.predict_proba(train2[val_idx,:], num_iteration=gbm.best_iteration_)[:,1]
# for cat in range(512):

#     train1 = train_df[(train_df[cat_names]==cat).values.reshape(-1,)]

#     test1 = test_df[(test_df[cat_names]==cat).values.reshape(-1,)]

#     idx = test1.index

#     idx1 = train1.index

#     # reset index for cv

#     train1.reset_index(drop=True, inplace=True)

#     # Do not reset test set's index

    

#     # feature selection

#     fs = pca.fit(train1[cont_names])

#     train2 = fs.transform(train1[cont_names])

#     test2 = fs.transform(test1[cont_names])

    

# #     fs = lda.fit(train1[cont_names], train1[dep_var])

# #     train2 = fs.transform(train1[cont_names])

# #     test2 = fs.transform(test1[cont_names])

# #     print(train2.shape, test2.shape)

    

   

#     # cv

#     folds, aucs = 10, []

#     cv = StratifiedKFold(n_splits=folds, random_state=42)

#     for train_idx, val_idx in cv.split(train2, train1[dep_var].values):

#         # make data

#         gbm = lgb.LGBMClassifier(**params)

#         _ = gbm.fit(train2[train_idx], train1[dep_var].values[train_idx], eval_set=[(train2[val_idx], train1[dep_var].values[val_idx])], 

#                     eval_metric=['binary_logloss', 'auc'], early_stopping_rounds=5, verbose=False)

        

#         # predict

#         preds = gbm.predict_proba(test2, num_iteration=gbm.best_iteration_)

#         preds_test[idx] += preds[:,1]/folds

        

#         # eval

#         preds_train[idx1[val_idx]] = gbm.predict_proba(train2[val_idx,:], num_iteration=gbm.best_iteration_)[:,1]
# roc_auc_score(train_df[dep_var], preds_train)
# submission['target'] = preds_test
# submission.to_csv('submission1.csv', index=None, encoding='utf-8')
# from catboost import CatBoostClassifier
# 构建模型

# config = {

#     'iterations': 2000,

#     'learning_rate': 1,

#     'custom_loss': ['AUC', 'Accuracy'],

#     'max_depth': 5,

#     'loss_function': 'Logloss',

#     'random_seed': 120,

#     'leaf_estimation_method': 'Gradient',

#     'l2_leaf_reg': 1e-3,

#     'max_leaves': 31

# }



# model = CatBoostClassifier(**config)



# # train

# model.fit(X_train, y_train, use_best_model=True, plot=True, 

#           early_stopping_rounds=15, cat_features=[0], eval_set=(X_valid, y_valid))
# preds_class = model.predict(X_valid, prediction_type='Class')

# # acc

# accuracy_score(y_valid, preds_class)
# preds = model.predict_proba(X_test)
# submission['target'] = preds[:, 1]
# submission.to_csv('submission3.csv', index=None, encoding='utf-8')
preds_test1 = np.zeros(len(test_df))

preds_train = np.zeros(len(train_df))

all_aucs = {}
for cat in range(512):

    train1 = train_df[(train_df[cat_names]==cat).values.reshape(-1,)]

    test1 = test_df[(test_df[cat_names]==cat).values.reshape(-1,)]

    idx = test1.index

    idx1 = train1.index

    

    # feature selection

    fs = VarianceThreshold(threshold=1.5).fit(train1[cont_names])

    train2 = fs.transform(train1[cont_names])

    test2 = fs.transform(test1[cont_names])

    

    # scale

    scaler = StandardScaler().fit(train2)

    train2 = scaler.transform(train2)

    test2 = scaler.transform(test2)

    

    cols = []

    for i in range(len(cont_names)):

        if fs.variances_[i] > 1.5:

            cols.append(cont_names[i])

    train3 = pd.DataFrame(train2, columns=cols)

    test3 = pd.DataFrame(test2, columns=cols)

    train3[dep_var] = train1[dep_var].values  # keep same index

    

    # reset index for cv

    train3.reset_index(drop=True, inplace=True)

    # Do not reset test set's index

    

    # cv

    folds, aucs = 10, []

    cv = StratifiedKFold(n_splits=folds, random_state=42)

    for train_idx, val_idx in cv.split(train2, train3[dep_var].values):

        # make data

        svnu = svm.NuSVC(probability=True, kernel='poly', degree=4, gamma='auto', random_state=123, nu=0.6);

        _ = svnu.fit(train2[train_idx], train3[dep_var].values[train_idx]);

        p1 = svnu.predict_proba(train2[val_idx]);

        auc = roc_auc_score(train3[dep_var].values[val_idx], p1[:, 1])

        aucs.append(auc)

#         print(auc)

        

        # predict

        preds = svnu.predict_proba(test2)

        preds_test1[idx] += preds[:,1]/folds

        

        # eval

        preds_train[idx1[val_idx]] = svnu.predict_proba(train2[val_idx,:])[:,1]

    all_aucs[cat] = np.mean(aucs)

#     print(f'mean auc {cat}', np.mean(aucs))

# print(sorted(all_aucs.items(), key=lambda x: x[1]))
# for cat in range(512):

#     train1 = train_df[(train_df[cat_names]==cat).values.reshape(-1,)]

#     test1 = test_df[(test_df[cat_names]==cat).values.reshape(-1,)]

#     idx = test1.index

#     idx1 = train1.index

#     # reset index for cv

#     train1.reset_index(drop=True, inplace=True)

#     # Do not reset test set's index

    

#     # feature selection

#     fs = pca.fit(train1[cont_names])

#     train2 = fs.transform(train1[cont_names])

#     test2 = fs.transform(test1[cont_names])

    

# #     fs = lda.fit(train1[cont_names], train1[dep_var])

# #     train2 = fs.transform(train1[cont_names])

# #     test2 = fs.transform(test1[cont_names])

# #     print(train2.shape, test2.shape)

    

#     # cv

#     folds, aucs = 10, []

#     cv = StratifiedKFold(n_splits=folds, random_state=42)

#     for train_idx, val_idx in cv.split(train2, train1[dep_var].values):

#         # make data

#         svnu = svm.NuSVC(probability=True, kernel='poly', degree=4, gamma='auto', random_state=123, nu=0.6);

#         _ = svnu.fit(train2[train_idx], train1[dep_var].values[train_idx]);

#         p1 = svnu.predict_proba(train2[val_idx]);

#         auc = roc_auc_score(train1[dep_var].values[val_idx], p1[:, 1])

#         aucs.append(auc)

        

#         # predict

#         preds = svnu.predict_proba(test2)

#         preds_test1[idx] += preds[:,1]/folds

        

#         # eval

#         preds_train[idx1[val_idx]] = svnu.predict_proba(train2[val_idx,:])[:,1]

#     all_aucs[cat] = np.mean(aucs)

# #     print(f'mean auc {cat}', np.mean(aucs))

# print(sorted(all_aucs.items(), key=lambda x: x[1]))
roc_auc_score(train_df[dep_var], preds_train)
# submission['target'] = preds_test1 * 0.6 + preds_test * 0.2 + preds_ * 0.2

# submission['target'] = preds_test1 * 0.7 + preds_test * 0.3

submission['target'] = preds_test1 * 0.8 + preds_ * 0.2
submission.to_csv('submission.csv', index=None, encoding='utf-8')