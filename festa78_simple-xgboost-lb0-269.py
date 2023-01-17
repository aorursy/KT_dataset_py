# load Python modules

import pandas as pd

import numpy as np

import warnings

import xgboost as xgb

warnings.filterwarnings('ignore')
### load data

train_df = pd.read_csv("../input/train.csv")

test_df = pd.read_csv("../input/test.csv")

print('train size:', train_df.shape)

print('test size:', test_df.shape)

print('have the same columns?', all(train_df.drop('target', axis=1).columns == test_df.columns))

train_df_org = train_df

test_df_org = test_df
###data cleansing

# remove duplicatives if exists

# wrt rows

train_df = train_df.drop_duplicates()

# wrt columns (get recursion error)

#train_df = train_df.T.drop_duplicates().T



rows = train_df.shape[0]

columns = train_df.shape[1]

print("rows: {0}, columns: {1}".format(rows, columns))

test_df.head()
# remove constant values

train_df = train_df.loc[:, (train_df != train_df.iloc[0]).any()]

test_df = test_df.loc[:, train_df.drop('target', axis=1).columns]
# fill nan by median

train_df = train_df.replace(-1, np.NaN)

test_df = test_df.replace(-1, np.NaN)

print('nan exists in train?:', train_df.isnull().any().any())

print('nan exists in test?:', test_df.isnull().any().any())

train_median = train_df.drop('target', axis=1).median()

train_df = train_df.fillna(train_median)

test_df = test_df.fillna(train_median)
# separate data

train_y = train_df.loc[:, 'target']

train_id = train_df.loc[:, 'id']

train_df = train_df.drop(['target', 'id'], axis=1)

train_df_float = train_df.select_dtypes(include=['float64'])

train_df_int = train_df.select_dtypes(include=['int64'])

test_id = test_df.loc[:, 'id']

test_df = test_df.drop('id', axis=1)

test_df_float = test_df.select_dtypes(include=['float64'])

test_df_int = test_df.select_dtypes(include=['int64'])

print('train float:', len(train_df_float.columns))

print('train int:', len(train_df_int.columns))

print('test float:', len(test_df_float.columns))

print('test int:', len(test_df_int.columns))
# normalize data

train_df_float_mean = train_df_float.mean()

train_df_float_std = train_df_float.std()

train_df_float_norm = (train_df_float - train_df_float_mean) / (train_df_float_std + 1.e-9)

test_df_float_norm = (test_df_float - train_df_float_mean) / (train_df_float_std + 1.e-9)



train_df_norm = pd.concat((train_df_float_norm, train_df_int), axis=1)

test_df_norm = pd.concat((test_df_float_norm, test_df_int), axis=1)

print(train_df_norm.shape)

print(test_df_norm.shape)
### learn xgboost w/ default parameters

xgb_model = xgb.XGBRegressor()

xgb_model.fit(train_df_norm, train_y)
predict_y = xgb_model.predict(test_df_norm)
predict_submit = pd.concat((test_id, pd.DataFrame(data=predict_y, columns=['target'])), axis=1)

predict_submit.head()
# save csv

predict_submit.to_csv('./xgb_submission.csv', index=False)