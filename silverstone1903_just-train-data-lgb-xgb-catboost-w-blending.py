import numpy as np 
import pandas as pd 
import os
import gc
import seaborn as sns
import matplotlib.pyplot as plt
import datetime
print(os.listdir("../input"))
train_df = pd.read_csv("../input/train.csv", parse_dates=["first_active_month"])
test_df = pd.read_csv("../input/test.csv", parse_dates=["first_active_month"])
train_df.head()
train_df["feature_1"].value_counts()
train_df["feature_2"].value_counts()
train_df["feature_3"].value_counts()
sns.set_style("ticks")
plt.figure(figsize=[6,4])
sns.distplot(train_df["target"])
plt.show()
train_df.isna().sum()
test_df.isna().sum()
train_df.shape
train_df.dtypes
# from old fastai old
# https://github.com/fastai/fastai/blob/master/old/fastai/structured.py#L76
import re
def add_datepart(df, fldname, drop=False, time=False):
    fld = df[fldname]
    fld_dtype = fld.dtype
    if isinstance(fld_dtype, pd.core.dtypes.dtypes.DatetimeTZDtype):
        fld_dtype = np.datetime64

    if not np.issubdtype(fld_dtype, np.datetime64):
        df[fldname] = fld = pd.to_datetime(fld, infer_datetime_format=True)
    targ_pre = re.sub('[Dd]ate$', '', fldname)
    attr = ['Year', 'Month', 'Week', 'Day', 'Dayofweek', 'Dayofyear',
            'Is_month_end', 'Is_month_start', 'Is_quarter_end', 'Is_quarter_start', 'Is_year_end', 'Is_year_start']
    if time: attr = attr + ['Hour', 'Minute', 'Second']
    for n in attr: df[targ_pre + n] = getattr(fld.dt, n.lower())
    df[targ_pre + 'Elapsed'] = fld.astype(np.int64) // 10 ** 9
    if drop: df.drop(fldname, axis=1, inplace=True)

add_datepart(train_df, "first_active_month")
add_datepart(test_df, "first_active_month")
train_df['elapsed_time'] = (datetime.date(2018, 2, 1) - train_df['first_active_month'].dt.date).dt.days
test_df['elapsed_time'] = (datetime.date(2018, 2, 1) - test_df['first_active_month'].dt.date).dt.days
print("Train Data Time Range:",train_df["first_active_month"].min(), "-", train_df["first_active_month"].max())
print("Test Data Time Range:",test_df["first_active_month"].min(), "-",  test_df["first_active_month"].max())
train_df.groupby("first_active_month")["target"].mean().plot()
set((train_df["first_active_month"])).intersection(set(test_df["first_active_month"]))
train_card_id = train_df["card_id"]
test_card_id = test_df["card_id"]
train_df.sort_values("first_active_month", inplace=True)
train_df.reset_index(drop=True, inplace=True)
train_columns = [c for c in train_df.columns if c not in ["first_active_month", "card_id", "target"]]
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(train_df[train_columns], train_df["target"], test_size = 0.2, random_state=1001)
(X_train.shape, y_train.shape), (X_val.shape, y_val.shape)
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
params = {
"objective" : "regression",
"metric" : "rmse",
"num_leaves" : 12,
"learning_rate" : 0.01,
"bagging_fraction" : 0.7,
"feature_fraction" : 0.9,
"bagging_frequency" : 4,
"bagging_seed" : 1001,
"verbosity" : -1
}
cat_cols = ["feature_1", "feature_2", "feature_3"]
dtrain = lgb.Dataset(X_train, label=y_train, categorical_feature=cat_cols, free_raw_data=False)
dval = lgb.Dataset(X_val, label=y_val, categorical_feature=cat_cols, free_raw_data=False)
evals_result = {}
model_lgb = lgb.train(params, dtrain, 500, valid_sets=[dtrain,dval], valid_names=["train", "val"], early_stopping_rounds=30, verbose_eval=100, evals_result=evals_result)
lgb.plot_importance(model_lgb, figsize=(12,10))
lgb.plot_importance(model_lgb, figsize=(12,10), importance_type="gain")
lgb.plot_metric(evals_result, "rmse", figsize=(12,10))
pred_lgb = model_lgb.predict(test_df[train_columns])
ss = pd.DataFrame({"card_id":test_card_id, "target":pred_lgb})
ss.to_csv("preds_starter_lgb.csv", index=None)
ss.head()
import xgboost as xgb
dtrain_xgb = xgb.DMatrix(X_train, label=y_train)
dval_xgb = xgb.DMatrix(X_val, y_val)
xgb_params = {
        'objective': 'reg:linear',
        'learning_rate': 0.02,
        'max_depth': 5,
        'subsample': 0.7,
        'colsample_bytree': 0.9,
        'colsample_bylevel': 0.9,
        'n_jobs': -1,
        "silent": 1
    }
evallist  = [(dtrain_xgb, 'train'), (dval_xgb, "val")]
model_xgb = xgb.train(xgb_params, dtrain_xgb, num_boost_round=500, evals=evallist, early_stopping_rounds=50, verbose_eval=100)
xgb.plot_importance(model_xgb)
preds_xgb = model_xgb.predict(xgb.DMatrix(test_df[train_columns]))
ss = pd.DataFrame({"card_id":test_card_id, "target":preds_xgb})
ss.to_csv("preds_starter_xgb.csv", index=None)
ss.head()
from catboost import CatBoostRegressor
model_cat = CatBoostRegressor(iterations=500,
                             learning_rate=0.02,
                             depth=6,
                             eval_metric='RMSE',
                             bagging_temperature = 0.9,
                             od_type='Iter',
                             metric_period = 100,
                             od_wait=50)
model_cat.fit(X_train, y_train,
             eval_set=(X_val,y_val),
             cat_features=np.array([0,1,2]),
             use_best_model=True,
             verbose=100)
preds_cat = model_cat.predict(test_df[train_columns])  
pd.DataFrame(model_cat.get_feature_importance(), index=X_train[train_columns].columns, columns=["FeatureImportance"]).sort_values("FeatureImportance", ascending=False).plot(kind="barh", legend=False, figsize=(12,10))
ss = pd.DataFrame({"card_id":test_card_id, "target":preds_cat})
ss.to_csv("preds_starter_cat.csv", index=None)
ss.head()
ss_blend = pd.DataFrame({"card_id":test_card_id, "target":((0.33 * preds_xgb) + (0.34 * pred_lgb)+ (0.33 * preds_cat))})
ss_blend.to_csv("preds_starter_blend.csv", index=None)
ss_blend.head()