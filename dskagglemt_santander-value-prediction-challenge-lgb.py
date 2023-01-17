import numpy as np 

import pandas as pd



import matplotlib.pyplot as plt

import seaborn as sns
train = pd.read_csv("../input/santander-value-prediction-challenge/train.csv")

test = pd.read_csv("../input/santander-value-prediction-challenge/test.csv")
unique_df = train.nunique().reset_index()

unique_df.columns = ["col_name", "unique_count"]

constant_df = unique_df[unique_df["unique_count"]==1]

constant_df.shape
X = train.drop(constant_df.col_name.tolist() + ["ID", "target"], axis=1)

y = np.log1p(train["target"].values) # Our Evaluation metric for the competition is RMSLE. So let us use log of the target variable to build our models.



test_2 = test.drop(constant_df.col_name.tolist() + ["ID"], axis=1)
from sklearn import preprocessing, model_selection, metrics

import lightgbm as lgb
def run_lgb(X_train, y_train, X_val, y_val, test_df):

    params = {

        "objective" : "regression",

        "metric" : "rmse",

        "num_leaves" : 30,

        "learning_rate" : 0.01,

        "bagging_fraction" : 0.7,

        "feature_fraction" : 0.7,

        "bagging_frequency" : 5,

        "bagging_seed" : 2018,

        "verbosity" : -1

    }

    

    lgtrain = lgb.Dataset(X_train, label = y_train)

    lgval   = lgb.Dataset(X_val,   label = y_val  )

    

    evals_result = {}

    

    model = lgb.train(params, lgtrain, 1000, valid_sets=[lgval], early_stopping_rounds=100, verbose_eval=200, evals_result=evals_result)

    

    pred_test = model.predict(test_df, num_iteration=model.best_iteration)

    

    return pred_test, model, evals_result
kf = model_selection.KFold(n_splits=5, shuffle=True, random_state=99)



pred_test_full = 0
for train_index, val_index in kf.split(X):

    X_train, X_val = X.loc[train_index,:], X.loc[val_index,:]

    y_train, y_val = y[train_index], y[val_index]

    pred_test, model, evals_result = run_lgb(X_train, y_train, X_val, y_val, test_2) 

    pred_test_full += pred_test
pred_test_full /= 5.

pred_test_full = np.expm1(pred_test_full)
### Feature Importance ###

fig, ax = plt.subplots(figsize=(12,18))

lgb.plot_importance(model, max_num_features=20, height=0.8, ax=ax)

ax.grid(False)

plt.title("LightGBM - Feature Importance", fontsize=15)

plt.show()
# Making a submission file #

sub_df = pd.DataFrame({"ID":test["ID"].values})

sub_df["target"] = pred_test_full

sub_df.to_csv("lgb_v1.csv", index=False)
sub_df.head()