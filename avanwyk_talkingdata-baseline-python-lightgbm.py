import pandas as pd
import numpy as np
train_df = pd.read_csv('../input/train.csv', parse_dates=['click_time', 'attributed_time'], nrows=1000000)
def encode_cyclical(frame, col, max_val):
    frame[col + '_sin'] = np.sin(2 * np.pi * frame[col]/max_val)
    frame[col + '_cos'] = np.cos(2 * np.pi * frame[col]/max_val)
    return frame

def create_click_aggregate(frame, name, idxs):
    aggregate = frame.groupby(by=idxs, as_index=False).click_time.count()
    aggregate = aggregate.rename(columns={'click_time': name})
    return frame.merge(aggregate, on=idxs)

def unique_values_by_ip(frame, value):
    n_values_by_ip = frame.groupby(by='ip')[value].nunique()
    frame.set_index('ip', inplace=True)
    frame['n_' + value] = n_values_by_ip
    frame.reset_index(inplace=True)
    return frame

def impute_features(df):
    df['click_hour'] = df['click_time'].dt.hour + df['click_time'].dt.minute / 60
    df['click_day'] = df['click_time'].dt.day
    df['click_month'] = df['click_time'].dt.month
    cyclical_features = [('click_hour', 24), ('click_day', 31), ('click_month', 12)]
    for f in cyclical_features:
        df = encode_cyclical(df, *f)
        
    df = create_click_aggregate(df, 'total_clicks', ['ip'])
    df = create_click_aggregate(df, 'clicks_in_day', ['ip', 'click_month', 'click_day'])
    df = create_click_aggregate(df, 'clicks_in_hour', ['ip', 'click_month', 'click_day', 'click_hour'])

    df = unique_values_by_ip(df, 'os')
    df = unique_values_by_ip(df, 'app')
    df = unique_values_by_ip(df, 'device')
    df = unique_values_by_ip(df, 'channel')
    return df
from sklearn.model_selection import train_test_split

features = ['ip', 'app', 'device', 'os', 'channel', 'click_hour_sin', 'click_hour_cos', 'click_day_sin',
                'click_day_cos', 'click_month_sin', 'click_month_cos', 'total_clicks', 'clicks_in_day', 'clicks_in_hour', 'n_os', 'n_app', 'n_device', 'n_channel']

def create_dataset(df, test=False):
    X = df[features].values
    
    if test:
        ids = df['click_id']
        return X, ids
    
    y = df['is_attributed'].values
    
    return X, y

def create_train_val_data(df):
    imputed_df = impute_features(df)
    X, y = create_dataset(imputed_df)
    return train_test_split(X, y, test_size=0.33)

def create_test_data(df):
    imputed_df = impute_features(df)
    X, ids = create_dataset(imputed_df, test=True)
    return X, ids
import lightgbm as lgb
from sklearn.utils import class_weight
X_train, X_val, y_train, y_val = create_train_val_data(train_df)
def weigh_instances(y):
    class_weights = class_weight.compute_class_weight('balanced', np.unique(y), y)
    y_weighted = y.copy().astype(float)
    y_weighted[y==0] = class_weights[0]
    y_weighted[y==1] = class_weights[1]
    return y_weighted
y_train_weights = weigh_instances(y_train)
y_val_weights = weigh_instances(y_val)
categorical_features = [idx for idx in range(0, 5)]
lgb_train = lgb.Dataset(X_train, y_train, weight=y_train_weights,
                        categorical_feature=categorical_features, free_raw_data=False)
lgb_val = lgb.Dataset(X_val, y_val, weight=y_val_weights, reference=lgb_train,
                       categorical_feature=categorical_features, free_raw_data=False)
gbm = None
params = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'binary_logloss',
    'learning_rate': 0.01,
    'num_leaves': 31,
    'max_depth': -1,
    'min_child_samples': 20,
    'max_bin': 255,
    'subsample': 0.6,
    'subsample_freq': 0,
    'colsample_bytree': 0.3,
    'min_child_weight': 5,
    'subsample_for_bin': 200000,
    'min_split_gain': 0,
    'reg_alpha': 0.99,
    'reg_lambda': 0.9,
    'nthread': 8,
    'verbose': 0
}
gbm = lgb.train(params,
                lgb_train,
                init_model=gbm,
                num_boost_round=40,
                valid_sets=lgb_val,
                feature_name=features)
