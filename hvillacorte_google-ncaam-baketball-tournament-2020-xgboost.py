# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import xgboost as xgb

import sklearn.metrics as sklm



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
ver = 7



def score_model(probs, threshold):

    return np.array([1 if x > threshold else 0 for x in probs])



def print_metrics(labels, probs, threshold):

    scores = score_model(probs, threshold)

    metrics = sklm.precision_recall_fscore_support(labels, scores)

    logloss = sklm.log_loss(labels, probs)

    conf = sklm.confusion_matrix(labels, scores)

    print('                 Confusion matrix')

    print('                 Score positive    Score negative')

    print('Actual positive    %6d' % conf[0,0] + '             %5d' % conf[0,1])

    print('Actual negative    %6d' % conf[1,0] + '             %5d' % conf[1,1])

    print('')

    print(f"logloss: {logloss}")

    print('Accuracy        %0.2f' % sklm.accuracy_score(labels, scores))

    print('AUC             %0.2f' % sklm.roc_auc_score(labels, probs))

    print('Macro precision %0.2f' % float((float(metrics[0][0]) + float(metrics[0][1]))/2.0))

    print('Macro recall    %0.2f' % float((float(metrics[1][0]) + float(metrics[1][1]))/2.0))

    print(' ')

    print('           Positive      Negative')

    print('Num case   %6d' % metrics[3][0] + '        %6d' % metrics[3][1])

    print('Precision  %6.2f' % metrics[0][0] + '        %6.2f' % metrics[0][1])

    print('Recall     %6.2f' % metrics[1][0] + '        %6.2f' % metrics[1][1])

    print('F1         %6.2f' % metrics[2][0] + '        %6.2f' % metrics[2][1])
df = pd.read_csv("../input/ncaam2020/df.csv")

dftest = pd.read_csv("../input/ncaam2020/dftest.csv")

dfsubmit = pd.read_csv("../input/ncaam2020/submission.csv")
label = "Pred"



features = ["t1_N_win_perc","t2_N_win_perc","t1_rank","t2_rank","t1_win_perc","t2_win_perc",

            "t1_outscore","t2_outscore","t1_outscored","t2_outscored",

            "t1_points_avg","t2_points_avg","t1_fg_perc","t2_fg_perc",

            "t1_fg3_perc","t2_fg3_perc","t1_ft_perc","t2_ft_perc",

            "t1_or_avg","t1_dr_avg","t2_or_avg","t2_dr_avg","t1_ast_avg","t2_ast_avg",

            "t1_stl_avg","t2_stl_avg","t1_blk_avg","t2_blk_avg"]



df[features].head(10)
filternum = 0

df = df.loc[(df["t1_games"] > filternum) & (df["t2_games"] > filternum)]

print(f"Shape: {df.shape}")
X_train = df[features]

X_test = dftest[features]

y_train = df[label]

y_test = dftest[label]



dtrain = xgb.DMatrix(X_train, label=y_train)

dtest = xgb.DMatrix(X_test, label=y_test)

dsubmit = xgb.DMatrix(dfsubmit[features])
num_round = 999

param = {"max_depth":9,

         "min_child_weight":2,

         "subsample":0.8,

         "colsample_bytree":0.6,

         "eta":0.01,

         "eval_metric":"logloss",

         "objective":"binary:logistic",

         "seed":1,

         "verbosity":1}
scores = xgb.cv(param,

                dtrain,

                num_boost_round=num_round,

                nfold=2,

                metrics="logloss",

                verbose_eval=True,

                seed=1)

scores.to_csv(f"scores_{ver}.csv")

scores
num_round = scores["test-logloss-mean"].idxmin() + 1

num_round
bst = xgb.train(param, dtrain, num_round)

# make prediction

preds = bst.predict(dtest)

print_metrics(y_test, preds, 0.5)
# Predict

preds_proba = bst.predict(dsubmit)

preds_class = score_model(preds_proba, 0.5)

print(f"class = {preds_class[:20]}")

print(f"proba = {preds_proba[:20]}")



df = pd.DataFrame({"ID":list(dfsubmit["ID"]),"Pred":[i for i in preds_proba],"Class":preds_class})



# Submit

df.to_csv(f"sumbit{ver}_full.csv", index=False)

df[["ID","Pred"]].to_csv(f"sumbit{ver}.csv", index=False)
df = pd.DataFrame(bst.get_score(), index=["Score"])

s = df.loc["Score"].sort_values(ascending=False)

df = pd.DataFrame(s)

df.to_csv(f"feature_importances_{ver}.csv")

df