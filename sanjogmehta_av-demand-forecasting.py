import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import lightgbm as lgb
from tqdm import tqdm
import gc

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
train = pd.read_csv("../input/janatahack-demand-forecasting-analytics-vidhya/train.csv")
test = pd.read_csv("../input/janatahack-demand-forecasting-analytics-vidhya/test.csv")
submission = pd.read_csv("../input/janatahack-demand-forecasting-analytics-vidhya/sample_submission.csv")

print("train shape: {}".format(train.shape))
print("test shape: {}".format(test.shape))
print("submission shape: {}".format(submission.shape))
train.groupby(['store_id', 'sku_id']).size().reset_index(name="Count")
test.groupby(['store_id', 'sku_id']).size().reset_index(name="Count")
fday = datetime(2013, 7, 16)
fday
train_test_df = train.append(test, ignore_index=True)
train_test_df.tail()
def data_preprocessing(df):

    df['week'] = pd.to_datetime(df['week'], dayfirst=True)
    df['base_total_ratio'] = df['base_price']/df['total_price']
    df['total_base_ratio'] = df['total_price']/df['base_price']
    df['is_featured_displayed'] = df['is_featured_sku'] * df['is_display_sku']
    numcols = ['total_price', 'base_price', 'base_total_ratio', 'total_base_ratio','is_featured_sku', 'is_display_sku', 'is_featured_displayed','units_sold']
    
    df['store_sku_id'] = df['store_id'].astype(str) + '_' + df['sku_id'].astype(str)
    catcols = ['store_sku_id', 'store_id', 'sku_id']
    
    for col in catcols:
        if col != "store_sku_id":
            df[col] = df[col].astype('category')
            df[col] = df[col].cat.codes.astype("int16")
            df[col] -= df[col].min()
    
    return df
def create_features(df):
    lags = [12]
    lag_cols = [f"lag_{lag}" for lag in lags ]
    for lag, lag_col in tqdm(zip(lags, lag_cols), total = len(lags)):
        df[lag_col] = df[["store_sku_id","units_sold"]].groupby("store_sku_id")["units_sold"].shift(lag)

    wins = [6, 12, 26, 52]
    for win in tqdm(wins):
        for lag,lag_col in tqdm(zip(lags, lag_cols), total = len(lags)):
            df[f"rmean_{lag}_{win}"] = df[["store_sku_id", lag_col]].groupby("store_sku_id")[lag_col].transform(lambda x : x.rolling(win).mean())
    
    date_features = {
        
        "weekday": "weekday",
        "weekofyear": "weekofyear",
        "month": "month",
        "quarter": "quarter",
        "year": "year",
        "mday": "day",
        "ime": "is_month_end",
        "ims": "is_month_start",
    }
    
    for date_feat_name, date_feat_func in date_features.items():
        if date_feat_name in df.columns:
            df[date_feat_name] = df[date_feat_name].astype("int16")
        else:
            df[date_feat_name] = getattr(df["week"].dt, date_feat_func).astype("int16")
def rmsle_lgbm(y_pred, data):

    y_true = np.array(data.get_label())
    score = np.sqrt(np.mean(np.power(np.log1p(y_true) - np.log1p(y_pred), 2)))

    return 'rmsle', score, False
train_test_df = data_preprocessing(train_test_df)
%time

create_features(train_test_df)
train_test_df.shape
train_df = train_test_df[train_test_df['units_sold'].isnull() != True]
test_df = train_test_df[train_test_df['units_sold'].isnull() == True]
train_df.dropna(inplace = True)
train_df.shape
categorical_features = ['store_id', 'sku_id', 'is_featured_sku', 'is_display_sku', 'is_featured_displayed']
useless_cols = ['record_ID', 'week', 'store_sku_id', 'units_sold']
train_cols = train_df.columns[~train_df.columns.isin(useless_cols)]
X_train = train_df[train_cols]
y_train = train_df['units_sold']
%%time

np.random.seed(42)

valid_inds = np.random.choice(X_train.index.values, 5000, replace = False)
train_inds = np.setdiff1d(X_train.index.values, valid_inds)

train_data = lgb.Dataset(X_train.loc[train_inds] , label = y_train.loc[train_inds], 
                         categorical_feature=categorical_features, free_raw_data=False)
valid_data = lgb.Dataset(X_train.loc[valid_inds], label = y_train.loc[valid_inds],
                         categorical_feature=categorical_features, free_raw_data=False) # This is a random sample, we're not gonna apply any time series train-test-split tricks here!
params = {
        "objective" : "tweedie",
        'tweedie_variance_power': 1.1,
#         "metric" :"rmse",
        "force_row_wise" : True,
        "learning_rate" : 0.07,
#         "sub_feature" : 0.8,
        "sub_row" : 0.75,
        "bagging_freq" : 1,
        "lambda_l2" : 0.1,
        "lambda_l1" : 0.1,
#         "nthread" : 4
        "metric": 'custom',
        'verbosity': 1,
        'num_iterations' : 5000,
#         'num_leaves': 64,
#         "min_data_in_leaf": 100,
#         'num_leaves': 2**11-1,
#         'min_data_in_leaf': 2**12-1,
        'feature_fraction': 0.5,
#         'max_bin': 100,
#         'max_depth': 8,
        'early_stopping_round': 20
}
%%time

m_lgb = lgb.train(params, train_data, valid_sets = [valid_data], verbose_eval=20, feval=rmsle_lgbm) 
m_lgb.save_model("model_rmsle.lgb")
for tdelta in range(0, 80, 7):
    day = fday + timedelta(days=tdelta)
    print(tdelta, day)
    tst = test_df.loc[test_df.week == day , train_cols]
    print('tst shape: {}'.format(tst.shape))
    test_df.loc[test_df.week == day, "units_sold"] = m_lgb.predict(tst)
submission_lgb = test_df[['record_ID', 'units_sold']]
submission_lgb.to_csv("lgb_lag_12_win_6_12_itr_2k_bin_default_lr_0.07_es_20.csv", index=False)