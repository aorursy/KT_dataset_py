import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import os



base_dir = '../input'
X_train = pd.read_csv(f'{base_dir}/X_train.csv').drop(['measurement_number', 'row_id'], axis=1)

y_train = pd.read_csv(f'{base_dir}/y_train.csv')

train_set = X_train.merge(y_train, on='series_id')

X_test = pd.read_csv(f'{base_dir}/X_test.csv').drop(['measurement_number', 'row_id'], axis=1)
from functools import reduce

def transform_features(df):

    df['orientation_sum'] = df['orientation_X'] + df['orientation_Y'] + df['orientation_Z'] + df['orientation_W']

    df['velocity_sum'] = df['angular_velocity_X'] + df['angular_velocity_Y'] + df['angular_velocity_Z']

    df['acc_sum'] = df['linear_acceleration_X'] + df['linear_acceleration_Y'] + df['linear_acceleration_Z']

    df['orientation_prod'] = df['orientation_X'] * df['orientation_Y'] * df['orientation_Z'] * df['orientation_W']

    df['velocity_prod'] = df['angular_velocity_X'] * df['angular_velocity_Y'] * df['angular_velocity_Z']

    df['acc_prod'] = df['linear_acceleration_X'] * df['linear_acceleration_Y'] * df['linear_acceleration_Z']

    dfs = []

    for t in [('min', lambda x: x.min()), ('max', lambda x: x.max()), ('mean', lambda x: x.mean()), ('std_dev', lambda x: x.std())]:

        name = t[0]

        agg_func = t[1]

        agg_df = df.groupby('series_id', as_index=False).apply(agg_func).drop('series_id', axis=1).reset_index().rename(index=str, columns={'index': 'series_id'})

        agg_df['series_id'] = agg_df['series_id'].astype(int)

        agg_df.columns = ['series_id'] + [str(col) + f'_{name}' for col in agg_df.columns if not col == 'series_id']

        dfs.append(agg_df)

    df = reduce(lambda left, right: pd.merge(left, right, on='series_id'), dfs)

    return df
features_df = transform_features(X_train)
target_df = train_set[['series_id','surface']].groupby('series_id', as_index=False).first()
X = features_df.drop('series_id', axis=1).values

feature_names = features_df.drop('series_id', axis=1).columns.values

y = target_df['surface'].values

y = y.reshape(y.shape[0])
import xgboost

from sklearn.model_selection import StratifiedKFold

from sklearn.model_selection import cross_val_score



#grid = {base_score=0.5, booster='gbtree', colsample_bylevel=1,

#       colsample_bytree=1, gamma=0, learning_rate=0.1, max_delta_step=0,

#        max_depth=3, min_child_weight=1, missing=None, n_estimators=100,

#       n_jobs=1, nthread=None, objective='binary:logistic', random_state=0,

#       reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,

#       silent=True, subsample=1}

grid = {'max_depth': 20, 'n_estimators': 200, 'n_jobs':4}

model = xgboost.XGBClassifier()

model.set_params(**grid)

kfold = StratifiedKFold(n_splits=10, random_state=7)

results = cross_val_score(model, X, y, cv=kfold)

print(f'Accuracy: {results.mean():.2%} ({results.std():.2%})')
from sklearn.metrics import accuracy_score

group_ids = train_set[['series_id','group_id']].groupby('series_id', as_index=False).first()

xv_set = features_df.merge(target_df, on='series_id').merge(group_ids, on='series_id')

acc_sum = 0

for split in np.array_split(xv_set['group_id'].unique(), 4):

    xv_test_set = xv_set[xv_set['group_id'].isin(split)]

    xv_train_set = xv_set[~xv_set['group_id'].isin(split)]

    model = xgboost.XGBClassifier()

    model.set_params(**grid)

    model.fit(xv_train_set.drop(['series_id', 'group_id', 'surface'], axis=1).values, xv_train_set['surface'].values)

    y_pred = model.predict(xv_test_set.drop(['series_id', 'group_id', 'surface'], axis=1).values)

    accuracy = accuracy_score(xv_test_set['surface'].values, y_pred)

    acc_sum += accuracy

    print(f"Accuracy = {accuracy}")

print(f"Final cv accuracy = {acc_sum / 4:.2%}")
model = xgboost.XGBClassifier()

model.set_params(**grid)

model.fit(X, y)
model.get_booster().feature_names = None
from xgboost import plot_importance

ax = xgboost.plot_importance(model)

fig = ax.figure

fig.set_size_inches(20, 32)
X_pred = transform_features(X_test).drop('series_id', axis=1).values

preds = model.predict(X_pred)
pd.DataFrame(preds)[0].value_counts()
submission = X_test[['series_id']].groupby('series_id', as_index=False).first()

submission['surface'] = preds
submission.to_csv('submission.csv', index=False)
pd.read_csv('submission.csv')