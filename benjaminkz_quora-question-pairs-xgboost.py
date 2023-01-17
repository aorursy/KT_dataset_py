import numpy as np

import pandas as pd

import os

import warnings

warnings.filterwarnings("ignore")



import xgboost as xgb

from xgboost.sklearn import XGBClassifier
train = pd.read_csv("../input/quora-question-pairs-feature-extraction-2/train.csv")

test = pd.read_csv("../input/quora-question-pairs-feature-extraction-2/test.csv")

trainlabel = pd.read_csv("../input/quora-question-pairs-feature-extraction-2/trainlabel.csv")
dtrain = xgb.DMatrix(train, label = trainlabel)
p = 0.369197853026293

pos_public = (0.55410 + np.log(1 - p)) / np.log((1 - p) / p)

pos_private = (0.55525 + np.log(1 - p)) / np.log((1 - p) / p)

average = (pos_public + pos_private) / 2

print (pos_public, pos_private, average)
w0 = average * (1 - p) / ((1 - average) * p)

print(w0)
w1 = average / p

w2 = (1 - average) / (1 - p)

print(w1, w2)
def weighted_log_loss(preds, dtrain):

    label = dtrain.get_label()

    return "weighted_logloss", -np.mean(w1 * label * np.log(preds) + w2 * (1 - label) * np.log(1 - preds))
params = {}

params["objective"] = "binary:logistic"

params["eval_metric"] = "logloss"

params["eta"] = 0.1

params["max_depth"] = 6

params["min_child_weight"] = 1

params["gamma"] = 0

params["subsample"] = 0.8

params["colsample_bytree"] = 0.9

params["scale_pos_weight"] = 0.3632

params["tree_method"] = "gpu_hist"  # 使用GPU加速的直方图算法

params['max_bin'] = 256



model1 = xgb.cv(params, dtrain, num_boost_round = 2000, nfold = 10, 

                feval = weighted_log_loss, early_stopping_rounds = 200, 

                verbose_eval = 50)
fix_params = {}

fix_params["objective"] = "binary:logistic"

fix_params["eval_metric"] = "logloss"

fix_params["eta"] = 0.1

fix_params["gamma"] = 0

fix_params["subsample"] = 0.8

fix_params["colsample_bytree"] = 0.9

fix_params["scale_pos_weight"] = 0.3632

fix_params["tree_method"] = "gpu_hist"

fix_params["max_bin"] = 256



evaluation_list = []

for depth in [5, 6]:

    for child_weight in [1, 2.5, 4]:

        params = {**fix_params, **{"max_depth": depth, "min_child_weight": child_weight}}

        evaluation = xgb.cv(params, dtrain, num_boost_round = 650, nfold = 6, 

                            feval = weighted_log_loss, early_stopping_rounds = 100)

        # evaluation记录了每一轮迭代的交叉验证结果

        evaluation_list.append(evaluation)

        

for depth in [7, 8]:

    for child_weight in [4, 5, 6]:

        params = {**fix_params, **{"max_depth": depth, "min_child_weight": child_weight}}

        evaluation = xgb.cv(params, dtrain, num_boost_round = 650, nfold = 6, 

                            feval = weighted_log_loss, early_stopping_rounds = 100)

        # evaluation记录了每一轮迭代的交叉验证结果

        evaluation_list.append(evaluation)



evaluation_panel = pd.DataFrame()

for evaluation in evaluation_list:

    # evaluation的最后一行即相应参数组合的结果

    evaluation_panel = pd.concat([evaluation_panel, evaluation.iloc[-1, :]], axis = 1)

evaluation_panel
fix_params = {}

fix_params["objective"] = "binary:logistic"

fix_params["eval_metric"] = "logloss"

fix_params["eta"] = 0.1

fix_params["gamma"] = 0

fix_params["subsample"] = 0.8

fix_params["colsample_bytree"] = 0.9

fix_params["scale_pos_weight"] = 0.3632

fix_params["tree_method"] = "gpu_hist"

fix_params["max_bin"] = 256



evaluation_list = []

for depth in [5, 6, 7]:

    for child_weight in [3, 3.5, 4, 4.5]:

        params = {**fix_params, **{"max_depth": depth, "min_child_weight": child_weight}}

        evaluation = xgb.cv(params, dtrain, num_boost_round = 650, nfold = 6, 

                            feval = weighted_log_loss, early_stopping_rounds = 100)

        evaluation_list.append(evaluation)



evaluation_panel = pd.DataFrame()

for evaluation in evaluation_list:

    evaluation_panel = pd.concat([evaluation_panel, evaluation.iloc[-1, :]], axis = 1)

evaluation_panel
fix_params = {}

fix_params["objective"] = "binary:logistic"

fix_params["eval_metric"] = "logloss"

fix_params["eta"] = 0.1

fix_params["gamma"] = 0

fix_params["subsample"] = 0.8

fix_params["colsample_bytree"] = 0.9

fix_params["scale_pos_weight"] = 0.3632

fix_params["tree_method"] = "gpu_hist"

fix_params["max_depth"] = 6

fix_params["min_child_weight"] = 4



evaluation_list = []

for bin in [200, 230, 256, 280]:

    params = {**fix_params, **{"max_bin": bin}}

    evaluation = xgb.cv(params, dtrain, num_boost_round = 650, nfold = 6, 

                        feval = weighted_log_loss, early_stopping_rounds = 100)

    evaluation_list.append(evaluation)



evaluation_panel = pd.DataFrame()

for evaluation in evaluation_list:

    evaluation_panel = pd.concat([evaluation_panel, evaluation.iloc[-1, :]], axis = 1)

evaluation_panel
fix_params = {}

fix_params["objective"] = "binary:logistic"

fix_params["eval_metric"] = "logloss"

fix_params["eta"] = 0.08

fix_params["gamma"] = 0

fix_params["subsample"] = 0.8

fix_params["colsample_bytree"] = 0.9

fix_params["scale_pos_weight"] = 0.3632

fix_params["tree_method"] = "gpu_hist"

fix_params["max_depth"] = 6

fix_params["min_child_weight"] = 3.5



evaluation_list = []

for bin in [220, 240, 270]:

    params = {**fix_params, **{"max_bin": bin}}

    evaluation = xgb.cv(params, dtrain, num_boost_round = 650, nfold = 6, 

                        feval = weighted_log_loss, early_stopping_rounds = 100)

    evaluation_list.append(evaluation)



evaluation_panel = pd.DataFrame()

for evaluation in evaluation_list:

    evaluation_panel = pd.concat([evaluation_panel, evaluation.iloc[-1, :]], axis = 1)

evaluation_panel
fix_params = {}

fix_params["objective"] = "binary:logistic"

fix_params["eval_metric"] = "logloss"

fix_params["eta"] = 0.1

fix_params["gamma"] = 0

fix_params["scale_pos_weight"] = 0.3632

fix_params["tree_method"] = "gpu_hist"

fix_params["max_depth"] = 6

fix_params["min_child_weight"] = 4

fix_params["max_bin"] = 256



evaluation_list = []

for row in [0.7, 0.8, 0.9]:

    for col in [0.7, 0.8, 0.9]:

        params = {**fix_params, **{"subsample": row, "colsample_bytree": col}}

        evaluation = xgb.cv(params, dtrain, num_boost_round = 650, nfold = 6, 

                            feval = weighted_log_loss, early_stopping_rounds = 100)

        evaluation_list.append(evaluation)



evaluation_panel = pd.DataFrame()

for evaluation in evaluation_list:

    evaluation_panel = pd.concat([evaluation_panel, evaluation.iloc[-1, :]], axis = 1)

evaluation_panel
fix_params = {}

fix_params["objective"] = "binary:logistic"

fix_params["eval_metric"] = "logloss"

fix_params["eta"] = 0.1

fix_params["gamma"] = 0

fix_params["scale_pos_weight"] = 0.3632

fix_params["tree_method"] = "gpu_hist"

fix_params["max_depth"] = 6

fix_params["min_child_weight"] = 4

fix_params["max_bin"] = 256



evaluation_list = []

for row in [0.75, 0.8, 0.85]:

    for col in [0.85, 0.9]:

        params = {**fix_params, **{"subsample": row, "colsample_bytree": col}}

        evaluation = xgb.cv(params, dtrain, num_boost_round = 650, nfold = 6, 

                            feval = weighted_log_loss, early_stopping_rounds = 100)

        evaluation_list.append(evaluation)



evaluation_panel = pd.DataFrame()

for evaluation in evaluation_list:

    evaluation_panel = pd.concat([evaluation_panel, evaluation.iloc[-1, :]], axis = 1)

evaluation_panel
params = {}

params["objective"] = "binary:logistic"

params["eval_metric"] = "logloss"

params["eta"] = 0.06

params["gamma"] = 0

params["scale_pos_weight"] = 0.3632

params["tree_method"] = "gpu_hist"

params["max_depth"] = 6

params["min_child_weight"] = 4

params["max_bin"] = 256

params["subsample"] = 0.8

params["colsample_bytree"] = 0.9



model6 = xgb.cv(params, dtrain, num_boost_round = 6000, nfold = 10, 

                feval = weighted_log_loss, early_stopping_rounds = 150, 

                verbose_eval = 50)
params = {}

params["objective"] = "binary:logistic"

params["eval_metric"] = "logloss"

params["eta"] = 0.04

params["gamma"] = 0

params["scale_pos_weight"] = 0.3632

params["tree_method"] = "gpu_hist"

params["max_depth"] = 6

params["min_child_weight"] = 4

params["max_bin"] = 256

params["subsample"] = 0.8

params["colsample_bytree"] = 0.9



model4 = xgb.cv(params, dtrain, num_boost_round = 6000, nfold = 10, 

                feval = weighted_log_loss, early_stopping_rounds = 150, 

                verbose_eval = 50)
params = {}

params["objective"] = "binary:logistic"

params["eval_metric"] = "logloss"

params["eta"] = 0.02

params["gamma"] = 0

params["scale_pos_weight"] = 0.3632

params["tree_method"] = "gpu_hist"

params["max_depth"] = 6

params["min_child_weight"] = 4

params["max_bin"] = 256

params["subsample"] = 0.8

params["colsample_bytree"] = 0.9



model2 = xgb.cv(params, dtrain, num_boost_round = 6000, nfold = 10, 

                feval = weighted_log_loss, early_stopping_rounds = 150, 

                verbose_eval = 50)
params = {}

params["objective"] = "binary:logistic"

params["eval_metric"] = "logloss"

params["eta"] = 0.02

params["gamma"] = 0

params["scale_pos_weight"] = 0.3632

params["tree_method"] = "gpu_hist"

params["max_depth"] = 6

params["min_child_weight"] = 4

params["max_bin"] = 256

params["subsample"] = 0.8

params["colsample_bytree"] = 0.9



dtest = xgb.DMatrix(test)



t = pd.read_csv("../input/quora-question-pairs/test.csv")
model = xgb.train(params, dtrain, num_boost_round = 3600)

prediction = model.predict(dtest)



sub = pd.DataFrame()

sub['test_id'] = t["test_id"]

sub['is_duplicate'] = prediction

sub.to_csv('submission3600.csv', index=False)
model = xgb.train(params, dtrain, num_boost_round = 3800)

prediction = model.predict(dtest)



sub = pd.DataFrame()

sub['test_id'] = t["test_id"]

sub['is_duplicate'] = prediction

sub.to_csv('submission3800.csv', index=False)
model = xgb.train(params, dtrain, num_boost_round = 4100)

prediction = model.predict(dtest)



sub = pd.DataFrame()

sub['test_id'] = t["test_id"]

sub['is_duplicate'] = prediction

sub.to_csv('submission4100.csv', index=False)