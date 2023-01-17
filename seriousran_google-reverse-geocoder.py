!pip install reverse_geocoder

import reverse_geocoder as revgc

revgc.search((47.4943, -122.24092))
import pandas as pd

import numpy as np

from sklearn.model_selection import KFold

from sklearn.preprocessing import LabelEncoder
df_train = pd.read_csv('../input/train.csv')

df_test  = pd.read_csv('../input/test.csv')
df_train['price'] = np.log1p(df_train['price'])



df_train = df_train.loc[df_train['id']!=8990]

df_train = df_train.loc[df_train['id']!=456]

df_train = df_train.loc[df_train['id']!=7259]

df_train = df_train.loc[df_train['id']!=2777]

df_train = df_train.loc[df_train['bedrooms']<9]



skew_columns = ['sqft_living', 'sqft_lot', 'sqft_above', 'sqft_basement']



for c in skew_columns:

    df_train[c] = np.log1p(df_train[c].values)

    df_test[c] = np.log1p(df_test[c].values)

    

for df in [df_train,df_test]:

    df['date'] = df['date'].apply(lambda x: x[0:8])

    df['yr_renovated'] = df['yr_renovated'].apply(lambda x: np.nan if x == 0 else x)

    df['yr_renovated'] = df['yr_renovated'].fillna(df['yr_built'])



for df in [df_train,df_test]:

    df['total_rooms'] = df['bedrooms'] + df['bathrooms']

    #df['grade_condition'] = df['grade'] * df['condition']

    df['sqft_ratio'] = df['sqft_living'] / df['sqft_lot']

    df['sqft_total_size'] = df['sqft_living'] + df['sqft_lot'] + df['sqft_above'] + df['sqft_basement']

    df['sqft_total15'] = df['sqft_living15'] + df['sqft_lot15'] 

    df['is_renovated'] = df['yr_renovated'] - df['yr_built']

    df['is_renovated'] = df['is_renovated'].apply(lambda x: 0 if x == 0 else 1)

    df['date'] = df['date'].astype('int')

    

df_train['per_price'] = df_train['price']/df_train['sqft_total_size']

zipcode_price = df_train.groupby(['zipcode'])['per_price'].agg({'mean','var'}).reset_index()

df_train = pd.merge(df_train,zipcode_price,how='left',on='zipcode')

df_test = pd.merge(df_test,zipcode_price,how='left',on='zipcode')

del df_train['per_price']



y_reg = df_train['price']

del df_train['price']

del df_train['id']

test_id = df_test['id']

del df_test['id']



train_columns = [c for c in df_train.columns if c not in ['id']]



import lightgbm as lgb

from sklearn.metrics import mean_squared_error



param = {'num_leaves': 32,

         'min_data_in_leaf': 30, 

         'objective':'regression',

         'max_depth': -1,

         'learning_rate': 0.015,

         "min_child_samples": 15,

         "boosting": "gbdt",

         "feature_fraction": 0.9,

         "bagging_freq": 1,

         "bagging_fraction": 0.7,

         "bagging_seed": 11,

         "metric": 'rmse',

         "lambda_l1": 0.1,

         "verbosity": 1,

         "nthread": 8,

         "random_state": 1}

         

#prepare fit model with cross-validation

folds = KFold(n_splits=5, shuffle=True, random_state=777)

oof = np.zeros(len(df_train))

predictions = np.zeros(len(df_test))

feature_importance_df = pd.DataFrame()



#run model

for fold_, (trn_idx, val_idx) in enumerate(folds.split(df_train)):

    print(str(fold_)+'-th fold')

    trn_data = lgb.Dataset(df_train.iloc[trn_idx][train_columns], label=y_reg.iloc[trn_idx])#, categorical_feature=categorical_feats)

    val_data = lgb.Dataset(df_train.iloc[val_idx][train_columns], label=y_reg.iloc[val_idx])#, categorical_feature=categorical_feats)



    num_round = 100000

    clf = lgb.train(param, trn_data, num_round, valid_sets = [trn_data, val_data], verbose_eval=1000, early_stopping_rounds = 1500)

    oof[val_idx] = clf.predict(df_train.iloc[val_idx][train_columns], num_iteration=clf.best_iteration)

    #feature importance

    fold_importance_df = pd.DataFrame()

    fold_importance_df["Feature"] = train_columns

    fold_importance_df["importance"] = clf.feature_importance()

    fold_importance_df["fold"] = fold_ + 1

    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

    #predictions

    predictions += clf.predict(df_test[train_columns], num_iteration=clf.best_iteration) / folds.n_splits

    print()

cv = np.sqrt(mean_squared_error(oof, y_reg))

print(cv)
df_train = pd.read_csv('../input/train.csv')

df_test  = pd.read_csv('../input/test.csv')
df_train['price'] = np.log1p(df_train['price'])



df_train = df_train.loc[df_train['id']!=8990]

df_train = df_train.loc[df_train['id']!=456]

df_train = df_train.loc[df_train['id']!=7259]

df_train = df_train.loc[df_train['id']!=2777]

df_train = df_train.loc[df_train['bedrooms']<9]



skew_columns = ['sqft_living', 'sqft_lot', 'sqft_above', 'sqft_basement']



for c in skew_columns:

    df_train[c] = np.log1p(df_train[c].values)

    df_test[c] = np.log1p(df_test[c].values)

    

for df in [df_train,df_test]:

    df['date'] = df['date'].apply(lambda x: x[0:8])

    df['yr_renovated'] = df['yr_renovated'].apply(lambda x: np.nan if x == 0 else x)

    df['yr_renovated'] = df['yr_renovated'].fillna(df['yr_built'])

    

for df in [df_train,df_test]:

    df['total_rooms'] = df['bedrooms'] + df['bathrooms']

    #df['grade_condition'] = df['grade'] * df['condition']

    df['sqft_ratio'] = df['sqft_living'] / df['sqft_lot']

    df['sqft_total_size'] = df['sqft_living'] + df['sqft_lot'] + df['sqft_above'] + df['sqft_basement']

    df['sqft_total15'] = df['sqft_living15'] + df['sqft_lot15'] 

    df['is_renovated'] = df['yr_renovated'] - df['yr_built']

    df['is_renovated'] = df['is_renovated'].apply(lambda x: 0 if x == 0 else 1)

    df['date'] = df['date'].astype('int')

    

df_train['per_price'] = df_train['price']/df_train['sqft_total_size']

zipcode_price = df_train.groupby(['zipcode'])['per_price'].agg({'mean','var'}).reset_index()

df_train = pd.merge(df_train,zipcode_price,how='left',on='zipcode')

df_test = pd.merge(df_test,zipcode_price,how='left',on='zipcode')

del df_train['per_price']





# ----------------------------------  added  -------------------------------





#df_train['country'] = df_train.apply(lambda row: revgc.search((row['lat'],row['long']))[0]['cc'],axis=1)

df_train['admin1'] = df_train.apply(lambda row: revgc.search((row['lat'],row['long']))[0]['admin1'],axis=1)

df_train['admin2'] = df_train.apply(lambda row: revgc.search((row['lat'],row['long']))[0]['admin2'],axis=1)

df_train['st_name'] = df_train.apply(lambda row: revgc.search((row['lat'],row['long']))[0]['name'],axis=1)



le_admin1 = LabelEncoder()

le_admin2 = LabelEncoder()

le_st = LabelEncoder()

#le.fit(df_train['country'].unique())

#df_train['country'] = le.transform(le.fit(df_train['country']))

le_admin1.fit(df_train['admin1'].unique())

df_train['admin1'] = le_admin1.transform(df_train['admin1'])

le_admin2.fit(df_train['admin2'].unique())

df_train['admin2'] = le_admin2.transform(df_train['admin2'])

le_st.fit(df_train['st_name'].unique())

df_train['st_name'] = le_st.transform(df_train['st_name'])





#df_test['country'] = df_test.apply(lambda row: revgc.search((row['lat'],row['long']))[0]['cc'],axis=1)

df_test['admin1'] = df_test.apply(lambda row: revgc.search((row['lat'],row['long']))[0]['admin1'],axis=1)

df_test['admin2'] = df_test.apply(lambda row: revgc.search((row['lat'],row['long']))[0]['admin2'],axis=1)

df_test['st_name'] = df_test.apply(lambda row: revgc.search((row['lat'],row['long']))[0]['name'],axis=1)



le_admin1 = LabelEncoder()

le_admin2 = LabelEncoder()

le_st = LabelEncoder()

#le.fit(df_test['country'].unique())

#df_test['country'] = le.transform(le.fit(df_test['country']))

le_admin1.fit(df_test['admin1'].unique())

df_test['admin1'] = le_admin1.transform(df_test['admin1'])

le_admin2.fit(df_test['admin2'].unique())

df_test['admin2'] = le_admin2.transform(df_test['admin2'])

le_st.fit(df_test['st_name'].unique())

df_test['st_name'] = le_st.transform(df_test['st_name'])





# ----------------------------------  added  -------------------------------





y_reg = df_train['price']

del df_train['price']

del df_train['id']

test_id = df_test['id']

del df_test['id']



train_columns = [c for c in df_train.columns if c not in ['id']]



import lightgbm as lgb

from sklearn.metrics import mean_squared_error



param = {'num_leaves': 32,

         'min_data_in_leaf': 30, 

         'objective':'regression',

         'max_depth': -1,

         'learning_rate': 0.015,

         "min_child_samples": 15,

         "boosting": "gbdt",

         "feature_fraction": 0.9,

         "bagging_freq": 1,

         "bagging_fraction": 0.7,

         "bagging_seed": 11,

         "metric": 'rmse',

         "lambda_l1": 0.1,

         "verbosity": 1,

         "nthread": 8,

         "random_state": 1}

         

#prepare fit model with cross-validation

folds = KFold(n_splits=5, shuffle=True, random_state=777)

oof = np.zeros(len(df_train))

predictions = np.zeros(len(df_test))

feature_importance_df = pd.DataFrame()



#run model

for fold_, (trn_idx, val_idx) in enumerate(folds.split(df_train)):

    print(str(fold_)+'-th fold')

    trn_data = lgb.Dataset(df_train.iloc[trn_idx][train_columns], label=y_reg.iloc[trn_idx])#, categorical_feature=categorical_feats)

    val_data = lgb.Dataset(df_train.iloc[val_idx][train_columns], label=y_reg.iloc[val_idx])#, categorical_feature=categorical_feats)



    num_round = 100000

    clf = lgb.train(param, trn_data, num_round, valid_sets = [trn_data, val_data], verbose_eval=1000, early_stopping_rounds = 1500)

    oof[val_idx] = clf.predict(df_train.iloc[val_idx][train_columns], num_iteration=clf.best_iteration)

    #feature importance

    fold_importance_df = pd.DataFrame()

    fold_importance_df["Feature"] = train_columns

    fold_importance_df["importance"] = clf.feature_importance()

    fold_importance_df["fold"] = fold_ + 1

    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

    #predictions

    predictions += clf.predict(df_test[train_columns], num_iteration=clf.best_iteration) / folds.n_splits

    print()

cv = np.sqrt(mean_squared_error(oof, y_reg))

print(cv)
submmission = pd.read_csv('../input/sample_submission.csv')

submmission['price'] = np.expm1(predictions)

submmission.to_csv('./submission.csv', index=False)