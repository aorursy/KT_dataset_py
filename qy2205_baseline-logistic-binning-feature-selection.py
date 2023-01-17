import pandas as pd

import numpy as np



import matplotlib.pyplot as plt



import statsmodels.api as sm



from sklearn.model_selection import StratifiedKFold

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import roc_auc_score

from sklearn.preprocessing import KBinsDiscretizer



%matplotlib inline
train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")
train_x = train.drop(['id', 'target'], axis = 1)

train_y = train['target']

test_x = test.drop(["id"], axis = 1)
def baseline_model(train_x, train_y, run_num = 10, fold = 5):

    train_result, test_result = [], []

    for i in range(run_num):

        # result list

        train_fold, test_fold = [], []

        # split dataset

        skf = StratifiedKFold(n_splits = fold, shuffle = True)

        fold_num = 1

        for train_index, valid_index in skf.split(train_x, train_y):

            # dataset

            X_train, X_valid = train_x.iloc[train_index], train_x.iloc[valid_index]

            y_train, y_valid = train_y.iloc[train_index], train_y.iloc[valid_index]

            # model

            reg = LogisticRegression(solver = "liblinear", penalty = "l2")

            reg.fit(X_train, y_train)

            y_train_pred = reg.predict(X_train)

            y_valid_pred = reg.predict(X_valid)

            # result AUC

            train_auc = roc_auc_score(y_train, y_train_pred)

            test_auc = roc_auc_score(y_valid, y_valid_pred)

            if i == 1:

                print("TRAIN Fold {0}, AUC score: {1}".format(fold_num, round(train_auc, 4)))

                print("TEST Fold {0}, AUC score: {1}".format(fold_num, round(test_auc, 4)))

            fold_num += 1

            train_fold.append(train_auc)

            test_fold.append(test_auc)

        train_result.append(train_fold)

        test_result.append(test_fold)

    return train_result, test_result
train_result, test_result = baseline_model(train_x = train_x, train_y = train_y, run_num = 10, fold = 5)
def model_result(train_result, test_result):

    base_test_re = pd.DataFrame(test_result).T

    base_test_re.index = ['fold {0}'.format(i) for i in range(5)]

    base_test_re.columns = ['run {0}'.format(i) for i in range(10)]

    base_train_re = pd.DataFrame(train_result).T

    base_train_re.index = ['fold {0}'.format(i) for i in range(5)]

    base_train_re.columns = ['run {0}'.format(i) for i in range(10)]

    return base_train_re, base_test_re

base_train_re, base_test_re = model_result(train_result, test_result)
base_train_re
base_test_re.round(3)
def binning(data, feature, n_bins):

    est = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='uniform')

    est.fit(data[feature].values)

    Xt = est.transform(data[feature].values)

    data[feature] = pd.DataFrame(Xt)

    return data
train_x_bin = binning(train_x, train_x.columns, n_bins = 15)

test_x_bin = binning(test_x, test_x.columns, n_bins = 15)
train_result_bin, test_result_bin = baseline_model(train_x_bin, train_y, run_num = 10, fold = 5)
base_train_re_bin, base_test_re_bin = model_result(train_result, test_result)
base_train_re_bin
base_test_re_bin.round(3)
sig_features = []

for each_feature in train.columns[2:]:

    X = train[each_feature]

    X = sm.add_constant(X)

    y = train.iloc[:,1]

    model = sm.OLS(y, X)

    result = model.fit()

    pvalue = result.pvalues[1]

    # using 90% significance level

    if pvalue <= 0.1:

        print("Feature {0}, p value is {1}".format(each_feature, round(pvalue, 3)))

        sig_features.append(each_feature)
train_x = train.drop(['id', 'target'], axis = 1)

train_y = train['target']
train_select_x = train_x[sig_features]

train_select_bin_x = train_x_bin[sig_features]
train_result_select, test_result_select = baseline_model(train_select_x, train_y, run_num = 10, fold = 5)

base_train_re_select, base_test_re_select = model_result(train_result_select, test_result_select)
base_train_re_select
base_test_re_select
train_result_bin_select, test_result_bin_elect = baseline_model(train_select_bin_x, train_y, run_num = 10, fold = 5)

base_train_re_bin_select, base_test_re_bin_select = model_result(train_result_bin_select, test_result_bin_elect)
base_train_re_bin_select
base_test_re_bin_select
train_select_bin_x = train_x_bin[sig_features]

test_select_bin_x = test_x_bin[sig_features]
# split dataset

skf = StratifiedKFold(n_splits = 5, shuffle = True)

fold_num = 1

y_test = np.zeros(len(test_select_bin_x))

for train_index, valid_index in skf.split(train_select_bin_x, train_y):

    # dataset

    X_train, X_valid = train_select_bin_x.iloc[train_index], train_select_bin_x.iloc[valid_index]

    y_train, y_valid = train_y.iloc[train_index], train_y.iloc[valid_index]

    # model

    reg = LogisticRegression(solver = "liblinear", penalty = "l2")

    reg.fit(X_train, y_train)

    y_train_pred = reg.predict(X_train)

    y_valid_pred = reg.predict(X_valid)

    # result AUC

    train_auc = roc_auc_score(y_train, y_train_pred)

    test_auc = roc_auc_score(y_valid, y_valid_pred)

    fold_num += 1

    # predict test set

    y_test_fold = reg.predict_proba(test_select_bin_x)[:, 1]

    y_test += y_test_fold

y_test = y_test/5
sub = pd.read_csv("../input/sample_submission.csv")

sub['target'] = y_test

sub.to_csv("submission_logit.csv", index = False)