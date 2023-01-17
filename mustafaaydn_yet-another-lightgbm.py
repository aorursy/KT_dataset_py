import itertools as it

import os



import lightgbm as lgb

import numpy as np

import pandas as pd



from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC
for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
train = pd.read_csv("../input/titanic/train.csv")

train_X = train.drop(columns="Survived")

train_y = train["Survived"]
train_X
train_X["Sex"].value_counts(dropna=False)
train_X["Sex"] = train_X["Sex"].replace("male", 0).replace("female", 1)

train_X["Sex"].value_counts(dropna=False)
train_X["Name"].value_counts(dropna=False)
train_X.drop(columns="Name", inplace=True)
train_X["Age"].value_counts(dropna=False)
train_X["Parch"].value_counts(dropna=0)
train_X["PassengerId"].value_counts(dropna=False)
train_X.drop(columns="PassengerId", inplace=True)
train_X["Pclass"].value_counts(dropna=False)
base = 5

inverter = 4

train_X["Pclass"] = base ** (inverter - train_X["Pclass"])

train_X["Pclass"].value_counts(dropna=False)
train_X["SibSp"].value_counts(dropna=False)
train_X["Ticket"].value_counts(dropna=False)
train_X.drop(columns="Ticket", inplace=True)
train_X["Fare"].value_counts(dropna=False)
train_X["Fare"].isna().sum()
train_X["Cabin"].value_counts(dropna=False).head(30)
train_X["Cabin"].isna().sum()
train_X["Cabin"] = train_X["Cabin"].str[0]

train_X["Cabin"].value_counts(dropna=False)
train_X["Embarked"].value_counts(dropna=False)
def fill_majority(df_part):

    for i in range(len(df_part.columns)):

        ser_i = df_part.iloc[:, i]

        ser_i = ser_i.fillna(ser_i.mode().item())

        df_part.iloc[:, i] = ser_i

    return df_part



def fill_proportion(df_part):

    for i in range(len(df_part.columns)):

        ser_i = df_part.iloc[:, i]

        ser_i_vc = ser_i.value_counts()

        total = len(ser_i[ser_i.notna()])

        dist = ser_i_vc / total

        candidates = ser_i_vc.index

        num_nans = ser_i.isna().sum()

        fillers = np.random.choice(candidates, num_nans, p=dist)

        ser_i[ser_i.isna()] = fillers

        df_part.iloc[:, i] = ser_i

    return df_part
train_X["Age"] = fill_proportion(train_X["Age"].to_frame())

assert train_X["Age"].isna().sum() == 0
train_X["Cabin"] = fill_proportion(train_X["Cabin"].to_frame())

assert train_X["Cabin"].isna().sum() == 0
train_X["Embarked"] = fill_majority(train_X["Embarked"].to_frame())

assert train_X["Embarked"].isna().sum() == 0
train_X
assert train_X.isna().sum().sum() == 0
train_X.dtypes
len(train_X["Cabin"].unique()), len(train_X["Embarked"].unique())
train_X_numeric = pd.get_dummies(train_X, drop_first=True)

train_X_numeric
cat_cols = train_X.select_dtypes("object").columns

train_X[cat_cols] = train_X[cat_cols].astype("category")
train_dataset = lgb.Dataset(train_X, label=train_y)
base_params = {

    # Main parameters

    "objective": "binary",

    "boosting_type": "gbdt",

    "num_trees": 100,

    "learning_rate": 0.05,

    "num_leaves": 31,

    "tree_learner": "serial",

    "num_threads": 0,

    "device_type": "cpu",

    "seed": 1284,

    

    # Learning control parameters

    "force_col_wise": False,

    "force_row_wise": True,

    "histogram_pool_size": -1,

    "max_depth": 5,                             # Increase to overfit

    "min_data_per_leaf": 20,                    #  Decrease to overfit

    "min_sum_hessian_per_leaf": 1e-3,           #  Decrease to overfit

    "bagging_fraction": 0.8,                    # Increase to overfit

    "bagging_freq": 5,                          # Increase to overfit

    "feature_fraction": 0.9,                    # Increase to overfit

    "feature_fraction_bynode": 1.0,

    "feature_fraction_seed": 1284,

    "extra_trees": False,

    "extra_seed": None,

    "early_stopping_rounds": 6,

    "first_metric_only": False,

    "max_leaf_output": -1,

    "lambda_l1": 0.0,

    "lambda_l2": 0.4,                           #  Decrease to overfit

    "min_gain_to_split": 0.0,

    

    # dart related

    "drop_rate": None,

    "max_drop": None,

    "skip_drop": None,

    "xgboost_dart_mode": False,

    "uniform_drop": False,

    "drop_seed": None,



    # goss-related

    "top_rate": 0.2,

    "other_rate": 0.1,



    "min_data_per_group": 100,

    "max_cat_threshold": 32,

    "cat_l2": 10.0,

    "cat_smooth": 10.0,

    "max_cat_to_onehot": 4,

    "top_k": None,

    "monotone_constraints": None,

    "monotone_constraints_method": None,

    "monotone_penalty": None,

    "feature_contri": None,

    "forcedsplits_filename": "",

    "path_smooth": 0,

    "interaction_constraints": "",

    "verbosity": 1,



    "max_bin": 255,

    "max_bin_by_feature": None,

    "min_data_in_bin": 4,

    "bin_construct_sample_cnt": 200_000,

    "data_random_seed": 1284,

    "is_enable_sparse": True,

    "enable_bundle": True,                    # EFB!

    "use_missing": False,

    "zero_as_missing": False,

    "feature_pre_filter": True,

    "pre_partition": False,



    # "categorical_feature": "",              # data already marked as "category" dtype

    "forcedbins_filename": "",



    "boost_from_average": True,

    "reg_sqrt": False,

    "alpha": None,                          # Huber or Quantile Regression



    "metric": "cross_entropy",

}
param_space = {

    "num_trees": [50, 100, 150, 200, 250, 400],

    "learning_rate": [0.001, 0.005, 0.01, 0.05, 0.1, 0.5],

    "num_leaves": [31, 63, 127],

    

    "max_depth": [5, 7, 9],                             

    "min_data_per_leaf": [10, 30, 60, 120, 300],                         



    "lambda_l2": [0.001, 0.01, 0.1, 0.4, 0.9],                          

    "max_bin": [7, 31, 127],

}
results = {}

num_folds = 5

# Evaluation metric is accuracy

eval_metric = "binary_error"



for i, hyper_values in enumerate(it.product(*param_space.values())):

    # Form the this iterations param config

    hyper_dict = dict(zip(param_space.keys(), hyper_values))

    hyper_params = {**base_params, **hyper_dict}

    

    # Run 5-fold cv and save its mean score

    res = lgb.cv(hyper_params, train_dataset, nfold=num_folds, metrics=eval_metric)

    results[tuple(hyper_dict.items())] = np.mean(np.array(res[f"{eval_metric}-mean"]))

    

    # Print the tried param config every 10 iters (rely on dicts being ordered 3.7+ (3.6+ if CPython))

    if not i % 10:

        print(list(results.items())[-1], "\n"*2)
best_params = sorted(results.keys(), key=results.get)[0]

best_score = results[best_params]                      
best_params, best_score
best_params = {**base_params, **dict(best_params)}



# Disable early stopping for we are done validating

del best_params["early_stopping_rounds"]

best_lgbm = lgb.train(best_params, train_dataset)
# Need to round; lgb gives probabilities

preds_in_sample = best_lgbm.predict(train_X).round()
acc = accuracy_score(train_y, preds_in_sample)

f1 = f1_score(train_y, preds_in_sample)

ra = roc_auc_score(train_y, preds_in_sample)
acc
f1
ra
lgb.plot_importance(best_lgbm, importance_type="split")
lgb.plot_importance(best_lgbm, importance_type="gain")
test_X = pd.read_csv("../input/titanic/test.csv")
test_X["Sex"] = test_X["Sex"].replace("male", 0).replace("female", 1)
test_X.drop(columns="Name", inplace=True)
test_X.drop(columns="PassengerId", inplace=True)
test_X["Pclass"] = base ** (inverter - test_X["Pclass"])
test_X.drop(columns="Ticket", inplace=True)
test_X["Cabin"] = test_X["Cabin"].str[0]
test_X["Age"] = fill_proportion(test_X["Age"].to_frame())

assert test_X["Age"].isna().sum() == 0
test_X["Cabin"] = fill_proportion(test_X["Cabin"].to_frame())

assert test_X["Cabin"].isna().sum() == 0
test_X["Embarked"] = fill_majority(train_X["Embarked"].to_frame())

assert test_X["Embarked"].isna().sum() == 0
test_X
cat_cols = test_X.select_dtypes("object").columns

test_X[cat_cols] = test_X[cat_cols].astype("category")
preds_out_sample = best_lgbm.predict(test_X).round()
preds_out_sample
subm_df = pd.read_csv("../input/titanic/gender_submission.csv")

subm_df
subm_df["Survived"] = preds_out_sample.astype(int)

subm_df
subm_df.to_csv("../output/submission.csv", index=False)