# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import random



import warnings



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from sklearn.metrics import mean_squared_error

from sklearn.preprocessing import LabelEncoder



import lightgbm as lgb



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
warnings.filterwarnings(action='once')
calendar = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/calendar.csv', parse_dates=['date'], index_col='date')

calendar.head(5)
calendar.fillna('0', inplace=True)



label_encoder = LabelEncoder()

label_cols = ['event_name_1', 'event_type_1', 'event_name_2', 'event_type_2']



# Apply label encoder 

for col in label_cols:

    calendar[col] = label_encoder.fit_transform(calendar[col])



calendar.head(5)
calendar['is_weekend'] = calendar['wday'].apply(lambda x: 1 if x == 1 or x == 2 else 0)

seasons = {1: 1, 2: 1, 12: 1, 3: 2, 4: 2, 5: 2, 6: 3, 7: 3, 8: 3, 9: 4, 10: 4, 11: 4 }

calendar['season'] = calendar['month'].apply(lambda x: seasons[x])

calendar.head()
calendar.info()
calendar['wm_yr_wk'] = calendar['wm_yr_wk'].astype(np.int16)

calendar['wday'] = calendar['wday'].astype(np.int8)

calendar['month'] = calendar['month'].astype(np.int8)

calendar['year'] = calendar['year'].astype(np.int16)

calendar['snap_CA'] = calendar['snap_CA'].astype(np.int8)

calendar['snap_TX'] = calendar['snap_TX'].astype(np.int8)

calendar['snap_WI'] = calendar['snap_WI'].astype(np.int8)

calendar['is_weekend'] = calendar['is_weekend'].astype(np.int8)

calendar['season'] = calendar['season'].astype(np.int8)

calendar['event_name_1'] = calendar['event_name_1'].astype(np.int16)

calendar['event_type_1'] = calendar['event_type_1'].astype(np.int8)

calendar['event_name_2'] = calendar['event_name_2'].astype(np.int8)

calendar['event_type_2'] = calendar['event_type_2'].astype(np.int8)
calendar.info()
train_data = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/sales_train_evaluation.csv')

train_data.head()
def melt_item_group(group):

    numcols = [f"d_{day}" for day in range(0,1941 + 1)]

    catcols = ['id', 'item_id', 'dept_id','store_id', 'cat_id', 'state_id']

    

    for day in range(1942, 1942 + 28):

        group[f"d_{day}"] = np.nan



    dt = pd.melt(

        group,

        id_vars = catcols,

        value_vars = [col for col in group.columns if col.startswith("d_")],

        var_name = "d",

        value_name = "sales"

    )

    return dt
prices = pd.read_csv('../input/m5-forecasting-accuracy/sell_prices.csv')

prices['wm_yr_wk'] = prices['wm_yr_wk'].astype(np.int16)

prices['sell_price'] = prices['sell_price'].astype(np.float32)

prices['wm_yr_wk'] = prices['wm_yr_wk'].astype(np.int16)

prices.set_index(['store_id', 'item_id', 'wm_yr_wk'], inplace=True)

prices = prices.sort_index()
prices.head()
def convert_categorical_columns(df, columns):

    for column in columns:

        df[column] = df[column].astype('category')

        df[column] = df[column].cat.codes

        df[column] = df[column].astype('category')
def get_dataset_simple(df, label_column):

    test_df = df.loc['d_1914':,:].copy()

    

    valid_df = df.loc['d_1914': 'd_1941',:].copy()

    

    train_df = df.loc[df.index[0]: 'd_1913',:].copy()

    train_df.loc[:,label_column] = train_df.loc[:,label_column]

    

    return train_df, valid_df, test_df
def get_dataset(df, label_column):

    test_df = df.loc['d_1914':,:].copy()

    

    df_shape = df.loc[:'d_1941',:].shape

    samples_idx = random.sample(range(df_shape[0]), int(0.2 * df_shape[0]))

    

    valid_idxs = samples_idx

    

    train_idxs = list(set(list(range(0, df_shape[0] ))) - set(samples_idx))

    

    return train_idxs, valid_idxs, test_df
evaluation_data = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/sales_train_evaluation.csv')



evaluation_data.loc[:,'id'] = evaluation_data['id'].apply(lambda x: x[:-10] + 'validation')

evaluation_data.set_index('id', inplace=True)

evaluation_data = evaluation_data.loc[:,'d_1914': 'd_1941'].copy()



def get_rmse(submission_validation):

    sub = submission_validation.set_index('id')

    error = mean_squared_error(sub.values, evaluation_data.loc[sub.index,:].values)

    return error
def split_prediction(prediction):

    columns = ['id'] + ['F%s' % i for i in range(1, 29)]

    prediction = prediction.reset_index().set_index('id')

    prediction_evaluation = prediction.loc[:,'d_1914':'d_1941'].reset_index().copy()

    prediction_evaluation.columns = columns

    

    prediction_validation = prediction.loc[:,'d_1942':].reset_index().copy()

    prediction_validation.columns = columns

    prediction_validation.loc[:, 'id'] = prediction_validation['id'].str[:-10] + 'validation'

    

    return prediction_validation, prediction_evaluation
def create_price_features(group_df, horizon=28):

    lag = horizon // 7

    group_df['sell_price_diff_shift_1'] = group_df.groupby('id')['sell_price'].transform(lambda x: x - x.shift(1)).astype(np.float32)

    group_df[f'sell_price_diff_shift_{horizon}'] = group_df.groupby('id')['sell_price'].transform(lambda x: x - x.shift(horizon)).astype(np.float32)

    group_df['sell_price_diff_rolling_7'] = group_df.groupby('id')['sell_price'].transform(lambda x: x - x.rolling(7).mean()).astype(np.float32)

    

    group_df[f'sell_price_diff_shift_{horizon}_shift_1'] = group_df.groupby('id')['sell_price_diff_shift_1'].transform(lambda x: x.shift(horizon)).astype(np.float32)

    group_df[f'sell_price_diff_shift_{horizon}_shift_{horizon}'] = group_df.groupby('id')[f'sell_price_diff_shift_{horizon}'].transform(lambda x: x.shift(horizon)).astype(np.float32)

    group_df[f'sell_price_diff_shift_{horizon}_rolling_7'] = group_df.groupby('id')['sell_price_diff_rolling_7'].transform(lambda x: x.shift(horizon)).astype(np.float32)

    

    group_df[f'sell_price_diff_rolling_7_diff_rolling_7_shift{horizon}'] = group_df.groupby('id')['sell_price_diff_rolling_7'].transform(lambda x: x - x.shift(horizon)).astype(np.float32)
def get_created_features(horizon=28):

    feature_columns = [

        'sell_price_diff_shift_1',

        'sell_price_diff_rolling_7',

        f'sell_price_diff_shift_{horizon}',

        f'sell_price_diff_shift_{horizon}_shift_1',

        f'sell_price_diff_shift_{horizon}_shift_{horizon}',

        f'sell_price_diff_shift_{horizon}_rolling_7', # ?

        f'sell_price_diff_rolling_7_diff_rolling_7_shift{horizon}',

    ]

    return feature_columns
%%time



idx_feature = ['id']

categorical_feature = [

    'dept_id' ,

    'state_id',

    #'wday',

    #'month',

    #'year',

    'event_name_1',

    'event_type_1',

    'snap_CA',

    'snap_TX',

    'snap_WI',

    'event_name_2',

    'event_type_2',

    'is_weekend',

    'season',

]

feature_columns = categorical_feature + [

    'sales_shift_28',

    'sales_mean_rolling_4_wday_shift_4',

    'sales_mean_rolling_4_wday_shift_8',

    'sell_price',

]

#train_generated_features = ['day_min', 'day_mean', 'day_max']



label_column = 'sales'



dummy_subm = pd.DataFrame(columns=['id'] + ['F%s' % i for i in range(1, 29)])

submission_validation = dummy_subm.copy()

submission_evaluation = dummy_subm.copy()



submission_validation_lgb = dummy_subm.copy()

submission_evaluation_lgb = dummy_subm.copy()



groups = train_data.groupby('store_id')

len_group = len(groups)



for i, (store_id, group) in enumerate(groups):

    print(store_id)



    group = melt_item_group(group)

    group_df = group.join(calendar.set_index('d'), on='d')

    del group



    group_df.sales.fillna(0, inplace=True)

    

    group_df = group_df.set_index(['d'])

    

    # add features for unique store

    group_df['sales_shift_28'] = group_df.groupby(['id'])['sales'].transform(lambda x: x.shift(28)).astype(np.float32)

    

    group_df['sales_mean_rolling_4_wday'] = group_df.groupby(['id', 'wday'])['sales'].transform(lambda x: x.rolling(4).mean()).astype(np.float32)

    group_df['sales_mean_rolling_4_wday_shift_4'] = group_df.groupby(['id', 'wday'])['sales_mean_rolling_4_wday'].transform(lambda x: x.shift(4)).astype(np.float32)

    group_df['sales_mean_rolling_4_wday_shift_8'] = group_df.groupby(['id', 'wday'])['sales_mean_rolling_4_wday'].transform(lambda x: x.shift(8)).astype(np.float32)



    # add prices

    print('add prices')

    group_df = group_df.join(

        prices.loc[store_id], on=['item_id', 'wm_yr_wk']

    )

    label_cols_prices = ['state_id', 'dept_id']

    convert_categorical_columns(group_df, label_cols_prices)

    

    create_price_features(group_df)

    feature_columns += get_created_features()

    

    # drop rows with na

    group_df.dropna(inplace=True)

    print('drop rows with na')



    train_idxs, valid_idxs, test_df = get_dataset(

        group_df[feature_columns + [label_column] + idx_feature],

        label_column=label_column

    )

    

    print('get_dataset_simple')

    #key = all_id[:-10] + 'validation'

    prediction = test_df.reset_index().set_index(['d', 'id',])['sales_mean_rolling_4_wday_shift_4']#.values.tolist()

    prediction = prediction.unstack(level=0)

    prediction_validation, prediction_evaluation = split_prediction(prediction)

    

    try:

        #lgb

        dtrain = lgb.Dataset(group_df.iloc[train_idxs][feature_columns], label=group_df.iloc[train_idxs][label_column], categorical_feature=categorical_feature)

        dvalid = lgb.Dataset(group_df.iloc[valid_idxs][feature_columns], label=group_df.iloc[valid_idxs][label_column], categorical_feature=categorical_feature)



        param = {

            'boosting_type': 'gbdt',

            'objective': 'tweedie',

            #'tweedie_variance_power': 1.1,

            'metric': 'rmse',

            'subsample': 0.5,

            'subsample_freq': 1,

            'learning_rate': 0.03,

            'num_leaves': 1024,

            'min_data_in_leaf': 1024,

            'feature_fraction': 0.2,

            'max_bin': 10,

            'boost_from_average': False,

            'verbose': -1,

            #'lambda_l1': 0.8,

            #'lambda_l2': 0,

            #'min_gain_to_split': 1.,

            #'min_sum_hessian_in_leaf': 1e-3,

        }

        # https://lightgbm.readthedocs.io/en/latest/index.html

        bst = lgb.train(param, dtrain, valid_sets=[dvalid], num_boost_round = 2000, early_stopping_rounds=30, verbose_eval=True, categorical_feature=categorical_feature)



        prediction_lgb = bst.predict(test_df.reset_index().set_index(['d', 'id',])[feature_columns])

        prediction_lgb_df = test_df.reset_index()[['d', 'id']].copy()

        prediction_lgb_df['prediction'] = prediction_lgb

        prediction_lgb_df = prediction_lgb_df.set_index(['d', 'id',]).unstack(level=0).reset_index()

        prediction_lgb_df.columns = ['id'] + ['d_{}'.format(i) for i in range(1914, 1970)]

        prediction_validation_lgb, prediction_evaluation_lgb = split_prediction(prediction_lgb_df)

    except Exception as e:

        print(e)

        prediction_validation_lgb = prediction_validation

        prediction_evaluation_lgb = prediction_evaluation

    finally:

        submission_validation_lgb = submission_validation_lgb.append(prediction_validation_lgb)

        submission_evaluation_lgb = submission_evaluation_lgb.append(prediction_evaluation_lgb)



    submission_validation = submission_validation.append(prediction_validation)

    submission_evaluation = submission_evaluation.append(prediction_evaluation)
submission = submission_validation.append(submission_evaluation)

submission.to_csv('/kaggle/working/submission_mean.csv', index=False)



submission_lgb = submission_validation_lgb.append(submission_evaluation_lgb)

submission_lgb.to_csv('/kaggle/working/submission_lgb.csv', index=False)