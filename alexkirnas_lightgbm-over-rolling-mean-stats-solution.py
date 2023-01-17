%matplotlib inline

import datetime

import pandas as pd

from tqdm import tqdm

from sklearn.metrics import mean_squared_error

import numpy as np

from tqdm import tqdm_notebook as tqdm

import lightgbm as lgb

from sklearn.neighbors import NearestNeighbors



def root_mean_squared_error(t,p):

    return np.sqrt(mean_squared_error(t,p))
train_raw = pd.read_csv("../input/train.csv", parse_dates=["Date"])

train_raw.set_index("Date", inplace=True)



test = pd.read_csv("../input/test.csv", parse_dates=["Date"])

test.set_index("Date", inplace=True)



n_hexagons = 319
def get_date_features(df):

    """Add new features to dataframe"""

    

    df['dayofweek'] = df.index.map(lambda x: x.dayofweek) 

    df['is_weekend'] = df.index.map(lambda x: x.dayofweek // 5)

    df['day'] = df.index.map(lambda x: x.day)

    df['week'] = df.index.map(lambda x: x.week)

    df['hour'] = df.index.map(lambda x: x.hour)



    return df



def get_holidays(df):

    holidays = pd.read_csv('../input/public_holidays.csv', parse_dates=['date']).date

    df['is_holiday'] = np.isin(df.index.date, holidays.dt.date)

    

    return df
data = pd.concat([train_raw, test])



lags = [31, 60, 90, 180, 360]

columns = [x for x in data.columns if 'hex' in x]

features = []



for lag in tqdm(lags):

    feat = data[columns].rolling(window=(lag * 3)).mean().shift(lag * 3 + 1)

    feat.rename(columns={x: x + f'_lag_{lag}' for x in columns}, inplace=True)

    features.append(feat)

features = pd.concat(features, axis=1)

train_raw = train_raw.reset_index()

train_raw = pd.melt(train_raw, id_vars=['Date'])

train_raw.rename(columns={'variable': 'hex', 'value': 'target'}, inplace=True)

train_raw['hex'] = train_raw.hex.map(lambda x: int(x[4:]))

train_raw.set_index('Date', inplace=True)



test = test.reset_index()

test = pd.melt(test, id_vars=['Date'])

test.rename(columns={'variable': 'hex', 'value': 'target'}, inplace=True)

test['hex'] = test.hex.map(lambda x: int(x[4:]))

test.set_index('Date', inplace=True)
train_raw = get_date_features(train_raw)

train_raw = get_holidays(train_raw)



test = get_date_features(test)

test = get_holidays(test)
reduced_train = []

reduced_test = []



knn_data = pd.read_csv('../input/hexagon_centers.csv')

nbrs = NearestNeighbors(n_neighbors=7, algorithm='ball_tree').fit(knn_data[['latitude', 'longitude']].values)  # 1 (hex itself) + 6 (neighobors)

distances, indices = nbrs.kneighbors(knn_data[['latitude', 'longitude']].values)



def reduce_n_regions(df, i):

    # Get relvelant features for 7 nearest hexagons

    cols = [x for x in df.columns if not x.startswith('hex_') or int(x[4:7]) in indices[i]] 

    df = hex_df[cols].copy()

    

    # Rename features: ord_0 (1st neasrest hex), ord_1 (2nd neasrest hex), ...

    df.rename(columns={x: f'ord_{np.where(indices[i] == int(x[4:7]))[0][0]}' + x[7:] 

                       for x in df.columns if x.startswith('hex_')}, inplace=True)

    

    return df



for i in tqdm(range(n_hexagons)):

    hex_df = train_raw[train_raw['hex'] == i].join(features)

    reduced_train.append(reduce_n_regions(hex_df, i))

    

    hex_df = test[test['hex'] == i].join(features)

    reduced_test.append(reduce_n_regions(hex_df, i))



del train_raw, test
import gc

gc.collect()
train_raw = pd.concat(reduced_train, sort=True)

test = pd.concat(reduced_test, sort=True)
start = datetime.datetime(2017,1, 1)

split_date = datetime.datetime(2018,10, 31)



val_true = train_raw[train_raw.index >= split_date]

train = train_raw[(train_raw.index >= start) & (train_raw.index < split_date)]
features = [x for x in train.columns if x not in ['target']]

trn_data = lgb.Dataset(train[features], label=train['target'])

val_data = lgb.Dataset(val_true[features], label=val_true['target'])





params = {

    'objective': 'regression',

    'metric': 'rmse',

    'seed': 0xCAFFE,

    

    'boosting': 'gbdt',

    'num_iterations': 1000,

    'early_stopping_round': 50,

    

    'max_depth': -1,

    'num_leaves': 31,

    

    'verbosity': 50,

}



clf = lgb.train(

    params, trn_data,

    valid_sets=[trn_data, val_data],

    verbose_eval=1,

)
predictions = clf.predict(test[features], num_iteration=clf.best_iteration)
submission  = pd.DataFrame({

    'hex': test.hex,

    'Incidents': predictions

}).reset_index()
submission.head()
final = []

for i, row in tqdm(submission.iterrows(), total=submission.shape[0]):

    final.append({

        'id': row.Date.strftime("%Y-%m-%d %X") + '_hex_%03d' % row.hex,

        'Incidents': row.Incidents,

    })

final = pd.DataFrame.from_dict(final)
final.head()
final.to_csv("submission.csv", index=False)