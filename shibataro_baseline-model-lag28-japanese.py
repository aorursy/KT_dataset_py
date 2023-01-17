!ls /kaggle/input/m5narroweddata/narrowed_data
# system
import sys
import os
import pickle
from datetime import datetime

# data manipulation
import numpy as np
import pandas as pd
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)

# visualization
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

# ML model
import lightgbm as lgb
# optunaでハイパラチューニングを行う場合は以下を用いる
#import optuna.integration.lightgbm as lgb
from sklearn import preprocessing, metrics
from sklearn.model_selection import TimeSeriesSplit

# gabarge collection
import gc

# path
print("CWD: " + os.getcwd())
DATA = '/kaggle/input/m5narroweddata/narrowed_data'
# メモリを節約する関数 (copied from https://www.kaggle.com/ratan123/m5-forecasting-lightgbm-with-timeseries-splits)
def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)    
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df
# 一旦学習データだけを読み込む
with open(DATA+'/sales_data.pkl', 'rb') as f:
    sales_data = pickle.load(f)
    
print(sales_data.shape)
sales_data.head()
# 実行時間短縮のために期間の絞り込み
MIN_DATE = datetime(2015, 5, 1)
sales_data = sales_data.loc[
    pd.to_datetime(sales_data['date'], format='%Y-%m-%d') >= MIN_DATE
]

# https://www.kaggle.com/shibataro/make-datamart-japaneseで作成したsubmissionの方は
# dept_id, cat_id, state_idがないため、dropする
# これも時間短縮のため
sales_data = sales_data.drop(['dept_id', 'cat_id', 'state_id'], axis=1)
print(sales_data.shape)
sales_data.head()
gc.collect()
# submissionデータを読み込む
with open(DATA+'/submission_data.pkl', 'rb') as f:
    submission_data = pickle.load(f)
    
print(submission_data.shape)
submission_data.head()
# 学習データとsubmissionデータをcat
sales_data['type'] = 'validation'
submission_data['type'] = 'evaluation'

data = pd.concat([sales_data, submission_data])
del sales_data, submission_data
data.head()
data.tail()
# 1. カテゴリカル変数のLabelEncoding
def cat_LabelEncoding(data):
    
    nan_features = ['event_name_1', 'event_type_1']
    for feature in nan_features:
        data[feature].fillna('unknown', inplace = True)
    
    encoder = preprocessing.LabelEncoder()
    data['id_encode'] = encoder.fit_transform(data['id'])
    
    cat = ['item_id', 'store_id', 'event_name_1', 'event_type_1']
    for feature in cat:
        encoder = preprocessing.LabelEncoder()
        data[feature] = encoder.fit_transform(data[feature])
    
    return data
data = cat_LabelEncoding(data)
gc.collect()
data.head()
# 2. ラグ特徴量の追加
def simple_fe(data):
    
    # demand features
    data['lag_t28'] = data.groupby(['id'])['demand'].transform(lambda x: x.shift(28))
    #data['lag_t29'] = data.groupby(['id'])['demand'].transform(lambda x: x.shift(29))
    #data['lag_t30'] = data.groupby(['id'])['demand'].transform(lambda x: x.shift(30))
    data['rolling_mean_t7'] = data.groupby(['id'])['demand'].transform(lambda x: x.shift(28).rolling(7).mean())
    data['rolling_std_t7'] = data.groupby(['id'])['demand'].transform(lambda x: x.shift(28).rolling(7).std())
    #data['rolling_mean_t30'] = data.groupby(['id'])['demand'].transform(lambda x: x.shift(28).rolling(30).mean())
    #data['rolling_mean_t90'] = data.groupby(['id'])['demand'].transform(lambda x: x.shift(28).rolling(90).mean())
    #data['rolling_mean_t180'] = data.groupby(['id'])['demand'].transform(lambda x: x.shift(28).rolling(180).mean())
    #data['rolling_std_t30'] = data.groupby(['id'])['demand'].transform(lambda x: x.shift(28).rolling(30).std())
    
    # price features
    #data['lag_price_t1'] = data.groupby(['id'])['sell_price'].transform(lambda x: x.shift(1))
    #data['price_change_t1'] = (data['lag_price_t1'] - data['sell_price']) / (data['lag_price_t1'])
    #data['rolling_price_max_t365'] = data.groupby(['id'])['sell_price'].transform(lambda x: x.shift(1).rolling(365).max())
    #data['price_change_t365'] = (data['rolling_price_max_t365'] - data['sell_price']) / (data['rolling_price_max_t365'])
    #data['rolling_price_std_t7'] = data.groupby(['id'])['sell_price'].transform(lambda x: x.rolling(7).std())
    #data['rolling_price_std_t30'] = data.groupby(['id'])['sell_price'].transform(lambda x: x.rolling(30).std())
    #data.drop(['rolling_price_max_t365', 'lag_price_t1'], inplace = True, axis = 1)
    
    # time features
    #data['date'] = pd.to_datetime(data['date'])
    #data['year'] = data['date'].dt.year
    #data['month'] = data['date'].dt.month
    #data['week'] = data['date'].dt.week
    #data['day'] = data['date'].dt.day
    #data['dayofweek'] = data['date'].dt.dayofweek
    
    return data
data = simple_fe(data)
data = reduce_mem_usage(data)
data.head()
# 学習データ作成
data = data.sort_values('date')
x = data.loc[data['type']=='validation'].drop(['demand', 'type'], axis=1)
y = data.loc[data['type']=='validation']['demand']
test = data.loc[data['type']=='evaluation'].drop(['demand', 'type'], axis=1)

del data
gc.collect()
print(x.shape)
x.head()
print(y.shape)
y.head()
print(test.shape)
test.head()
# クロスバリデーションの準備
n_fold = 2 # 時間短縮のため、2 fold CV
folds = TimeSeriesSplit(n_splits=n_fold)
splits = folds.split(x, y)
#学習用のパラメータ設定
params = {
    'num_leaves': 555,
    'min_child_weight': 0.034,
    'feature_fraction': 0.379,
    'bagging_fraction': 0.418,
    'min_data_in_leaf': 106,
    'objective': 'regression',
    'metrics': 'rmse', # 'regression'の場合のデフォルトなので、本来は不要
    'max_depth': -1,
    'learning_rate': 0.005,
    "boosting_type": "gbdt",
    "bagging_seed": 11,
    "metric": 'rmse',
    "verbosity": -1,
    'reg_alpha': 0.3899,
    'reg_lambda': 0.648,
    'random_state': 222,
}
# 学習と予測の実行
# 予測は各fold毎に行い、最終的な予測値はそれらの平均値として算出
# ついでにfeature importanceも算出

# 学習、予測に用いるカラムの指定
columns = [
    'item_id', 'store_id',
    'event_name_1', 'event_type_1',
    'snap_CA', 'snap_TX', 'snap_WI', 'sell_price',
    'lag_t28', 'rolling_mean_t7', 'rolling_std_t7'
]

# 予測値、スコアの入れ物
y_preds = np.zeros(test.shape[0])
y_oof = np.zeros(x.shape[0])
mean_score = []
# feature importanceの入れ物
feature_importances = pd.DataFrame()
feature_importances['feature'] = columns

# 学習と予測
for fold_n, (train_index, valid_index) in enumerate(splits):
    print('Fold:',fold_n+1)
    X_train, X_valid = x[columns].iloc[train_index], x[columns].iloc[valid_index]
    y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]
    
    dtrain = lgb.Dataset(X_train, label=y_train)
    dvalid = lgb.Dataset(X_valid, label=y_valid)
    
    clf = lgb.train(
        params,
        dtrain,
        500, # 時間短縮のため500roundで。多分足りない。
        valid_sets=[dtrain, dvalid],
        early_stopping_rounds=50,
        verbose_eval=100
    )
    
    # 各foldのfeature importance算出
    feature_importances['fold_{}'.format(fold_n+1)] = clf.feature_importance()
    
    # validationデータに対する予測
    y_pred_valid = clf.predict(X_valid,num_iteration=clf.best_iteration)
    
    # スコア(RMSE)の算出
    y_oof[valid_index] = y_pred_valid
    val_score = np.sqrt(metrics.mean_squared_error(y_pred_valid, y_valid))
    print(f'val rmse score is {val_score}')
    mean_score.append(val_score)
    
    # submissionデータに対する予測
    y_preds += clf.predict(test[columns], num_iteration=clf.best_iteration) / n_fold
    
    del X_train, X_valid, y_train, y_valid
    gc.collect()

print('mean rmse score over folds is',np.mean(mean_score))
test['demand'] = y_preds
# submissionのフォーマットへの修正
def sub_formatting(test, sub):
    predictions = test[['id', 'date', 'demand']]
    predictions = pd.pivot(predictions, index='id', columns='date', values='demand').reset_index()
    
    pred_eva = predictions.loc[predictions['id'].str.contains('evaluation')]\
                      .iloc[:, [0]+list(range(29, 57))]
    pred_val = predictions.loc[predictions['id'].str.contains('validation')]\
                      .iloc[:, :29]

    pred_eva.columns = ['id'] + ['F' + str(i+1) for i in range(28)]
    pred_val.columns = ['id'] + ['F' + str(i+1) for i in range(28)]

    predictions = pd.concat([pred_val, pred_eva]).reset_index(drop=True)
    predictions = pd.merge(sub[['id']], predictions, how='left', on='id')
    return predictions

# 提出用ファイル作成のためにsample_submission.csvの読み込み。必要なIDカラムだけ読み込む。
sub = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/sample_submission.csv')

subs = sub_formatting(test, sub)
subs.to_csv('submission.csv', index=False)
subs.head()
subs.tail()
#feature_importances['average'] = feature_importances[[f'fold_{fold_n + 1}' for fold_n in range(folds.n_splits)]].mean(axis=1)
feature_importances['average'] = feature_importances[[f'fold_{fold_n + 1}' for fold_n in range(2)]].mean(axis=1)
feature_importances.to_csv('feature_importances.csv')

plt.figure(figsize=(16, 12))
sns.barplot(data=feature_importances.sort_values(by='average', ascending=False).head(20), x='average', y='feature');
plt.title('20 TOP feature importance over {} folds average'.format(folds.n_splits));
