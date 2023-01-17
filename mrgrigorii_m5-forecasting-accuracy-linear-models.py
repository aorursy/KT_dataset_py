# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import random



import warnings



import category_encoders as ce



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from sklearn.metrics import mean_squared_error

from sklearn.preprocessing import LabelEncoder

from sklearn.linear_model import SGDRegressor, LinearRegression

from sklearn.preprocessing import StandardScaler



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
calendar.fillna('Regular', inplace=True)



label_encoder = LabelEncoder()

label_cols = ['event_name_1', 'event_type_1', 'event_name_2', 'event_type_2']



# Apply label encoder 

for col in label_cols:

    calendar[col] = label_encoder.fit_transform(calendar[col])
calendar['is_weekend'] = calendar['wday'].apply(lambda x: 1 if x == 1 or x == 2 else 0)

seasons = {1: 1, 2: 1, 12: 1, 3: 2, 4: 2, 5: 2, 6: 3, 7: 3, 8: 3, 9: 4, 10: 4, 11: 4 }

calendar['season'] = calendar['month'].apply(lambda x: seasons[x])

calendar.head()
train_data = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/sales_train_evaluation.csv')

train_data.head()
prices = pd.read_csv('../input/m5-forecasting-accuracy/sell_prices.csv')

prices['wm_yr_wk'] = prices['wm_yr_wk'].astype(np.int16)

prices['sell_price'] = prices['sell_price'].astype(np.float32)

prices['wm_yr_wk'] = prices['wm_yr_wk'].astype(np.int16)

prices.set_index(['store_id', 'item_id', 'wm_yr_wk'], inplace=True)

prices = prices.sort_index()

prices
def get_dataset_simple(df, label_column):

    test_df = df.loc['d_1914':,:].copy()

    

    valid_models_df = df.loc['d_1914': 'd_1941',:].copy()

    

    _df = df.loc[df.index[0]: 'd_1941',:].copy()

    samples_idx = random.sample(range(_df.shape[0]), int(0.2 * _df.shape[0]))

    

    valid_df = _df.iloc[samples_idx]

    

    idx_train = list(set(list(range(0, item_df.shape[0] ))) - set(samples_idx))

    train_df = df.iloc[idx_train].copy()

    #train_df.loc[:,label_column] = train_df.loc[:,label_column]

    

    return train_df, valid_df, test_df, valid_models_df
evaluation_data = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/sales_train_evaluation.csv')



evaluation_data.loc[:,'id'] = evaluation_data['id'].apply(lambda x: x[:-10] + 'validation')

evaluation_data.set_index('id', inplace=True)

evaluation_data = evaluation_data.loc[:,'d_1914': 'd_1941'].copy()



def get_rmse(submission_validation_boost):

    sub = submission_validation_boost.set_index('id')

    error = mean_squared_error(sub.values, evaluation_data.loc[sub.index,:].values)

    return error
def apply_target_encoder(item_df, cat_features_to_encoding):

    # TargetEncoder CatBoostEncoder

    target_enc = ce.TargetEncoder(cols=cat_features_to_encoding)

    target_enc.fit(item_df.loc[:'d_1913'][cat_features_to_encoding], item_df.loc[:'d_1913']['sales'])



    return item_df.join(target_enc.transform(item_df[cat_features_to_encoding]).add_suffix('_target'))
def create_sales_features(item_df, horizon=7):

    lag = horizon // 7

    item_df[f'sales_shift_{horizon}'] = item_df['sales'].shift(horizon)



    item_df[f'sales_shift_{horizon}_shift_{horizon}'] = item_df['sales'].shift(horizon * 2)

    

    item_df['sales_mean_rolling_4_wday'] = item_df.groupby(['wday'])['sales'].transform(lambda x: x.rolling(4).mean())

    item_df[f'sales_mean_rolling_4_wday_shift_{lag}'] = item_df.groupby(['wday'])['sales_mean_rolling_4_wday'].transform(lambda x: x.shift(lag))

    

    item_df[f'sales_mean_rolling_shift_{lag}_4_wday_shift_{lag}'] = item_df.groupby(['wday'])['sales_mean_rolling_4_wday'].transform(lambda x: x.shift(lag * 2))

    item_df[f'sales_diff_rolling_shift_{lag}_4_wday_shift_{lag}'] = item_df[f'sales_mean_rolling_4_wday_shift_{lag}'] - item_df[f'sales_mean_rolling_shift_{lag}_4_wday_shift_{lag}']
def create_price_features(item_df, horizon=7):

    lag = horizon // 7

    item_df['sell_price_diff_shift_1'] = item_df['sell_price'] - item_df['sell_price'].shift(1)

    item_df[f'sell_price_diff_shift_{horizon}'] = item_df['sell_price'] - item_df['sell_price'].shift(horizon)

    item_df['sell_price_diff_rolling_7'] = item_df['sell_price'] - item_df['sell_price'].rolling(7).mean()

    

    item_df[f'sell_price_diff_shift_{horizon}_shift_1'] = item_df['sell_price'].shift(horizon) - item_df['sell_price'].shift(horizon + 1)

    item_df[f'sell_price_diff_shift_{horizon}_shift_{horizon}'] = item_df['sell_price'].shift(horizon) - item_df['sell_price'].shift(horizon * 2)

    item_df[f'sell_price_diff_shift_{horizon}_rolling_7'] = item_df['sell_price'].shift(28) - item_df['sell_price'].shift(horizon).rolling(7).mean()

    

    item_df[f'sell_price_diff_rolling_7_diff_rolling_7_shift{horizon}'] = item_df['sell_price_diff_rolling_7'] - item_df['sell_price_diff_rolling_7'].shift(horizon)
def create_categorical_features(item_df, horizon=7):

    item_df[f'month_target_diff_{horizon}'] = item_df['month_target'] - item_df['month_target'].shift(horizon)

    item_df[f'season_target_diff_{horizon}'] = item_df['season_target'] - item_df['season_target'].shift(horizon)

    item_df[f'event_name_1_target_diff_{horizon}'] = item_df['event_name_1_target'] - item_df['event_name_1_target'].shift(horizon)

    item_df[f'event_type_1_target_diff_{horizon}'] = item_df['event_type_1_target'] - item_df['event_type_1_target'].shift(horizon)

    item_df[f'event_name_2_target_diff_{horizon}'] = item_df['event_name_2_target'] - item_df['event_name_2_target'].shift(horizon)

    item_df[f'event_type_2_target_diff_{horizon}'] = item_df['event_type_2_target'] - item_df['event_type_2_target'].shift(horizon)

    item_df[f'snap_CA_target_diff_{horizon}'] = item_df['snap_CA_target'] - item_df['snap_CA_target'].shift(horizon)

    item_df[f'snap_TX_target_diff_{horizon}'] = item_df['snap_TX_target'] - item_df['snap_TX_target'].shift(horizon)

    item_df[f'snap_WI_target_diff_{horizon}'] = item_df['snap_WI_target'] - item_df['snap_WI_target'].shift(horizon)

    

    item_df[f'season_target_diff_{horizon}_shift{horizon}'] = item_df['season_target'].shift(horizon) - item_df['season_target'].shift(horizon)

    item_df[f'event_name_1_target_diff_{horizon}_shift{horizon}'] = item_df['event_name_1_target'].shift(horizon) - item_df['event_name_1_target'].shift(horizon * 2)

    item_df[f'event_type_1_target_diff_{horizon}_shift{horizon}'] = item_df['event_type_1_target'].shift(horizon) - item_df['event_type_1_target'].shift(horizon * 2)

    item_df[f'event_name_2_target_diff_{horizon}_shift{horizon}'] = item_df['event_name_2_target'].shift(horizon) - item_df['event_name_2_target'].shift(horizon * 2)

    item_df[f'event_type_2_target_diff_{horizon}_shift{horizon}'] = item_df['event_type_2_target'].shift(horizon) - item_df['event_type_2_target'].shift(horizon * 2)

    item_df[f'snap_CA_target_diff_{horizon}_shift{horizon}'] = item_df['snap_CA_target'].shift(horizon) - item_df['snap_CA_target'].shift(horizon * 2)

    item_df[f'snap_TX_target_diff_{horizon}_shift{horizon}'] = item_df['snap_TX_target'].shift(horizon) - item_df['snap_TX_target'].shift(horizon * 2)

    item_df[f'snap_WI_target_diff_{horizon}_shift{horizon}'] = item_df['snap_WI_target'].shift(horizon) - item_df['snap_WI_target'].shift(horizon * 2)
def get_created_features(horizon=7):

    lag = horizon // 7

    

    linear_feature_columns = [

        #f'sales_shift_{horizon}',

        f'sales_mean_rolling_4_wday_shift_{lag}',

        f'sell_price_diff_rolling_7_diff_rolling_7_shift{horizon}',

        

        f'month_target_diff_{horizon}',

        f'season_target_diff_{horizon}',

        f'event_name_1_target_diff_{horizon}',

        f'event_type_1_target_diff_{horizon}',

        f'event_name_2_target_diff_{horizon}',

        f'event_type_2_target_diff_{horizon}',

        f'snap_CA_target_diff_{horizon}',

        f'snap_TX_target_diff_{horizon}',

        f'snap_WI_target_diff_{horizon}',

    ]

    

    feature_columns = [

        #f'sales_shift_{horizon}',

        #f'sales_shift_{horizon}_shift_{horizon}',

        f'sales_mean_rolling_4_wday_shift_{lag}',

        f'sales_mean_rolling_shift_{lag}_4_wday_shift_{lag}',

        f'sales_diff_rolling_shift_{lag}_4_wday_shift_{lag}',

        

        'sell_price_diff_shift_1',

        'sell_price_diff_rolling_7',

        f'sell_price_diff_shift_{horizon}',

        f'sell_price_diff_shift_{horizon}_shift_1',

        f'sell_price_diff_shift_{horizon}_shift_{horizon}',

        f'sell_price_diff_shift_{horizon}_rolling_7', # ?

        f'sell_price_diff_rolling_7_diff_rolling_7_shift{horizon}',

        

        f'month_target_diff_{horizon}',

        f'season_target_diff_{horizon}',

        f'event_name_1_target_diff_{horizon}',

        f'event_type_1_target_diff_{horizon}',

        f'event_name_2_target_diff_{horizon}',

        f'event_type_2_target_diff_{horizon}',

        f'snap_CA_target_diff_{horizon}',

        f'snap_TX_target_diff_{horizon}',

        f'snap_WI_target_diff_{horizon}',

        

        f'season_target_diff_{horizon}_shift{horizon}',

        f'event_name_1_target_diff_{horizon}_shift{horizon}',

        f'event_type_1_target_diff_{horizon}_shift{horizon}',

        f'event_name_2_target_diff_{horizon}_shift{horizon}',

        f'event_type_2_target_diff_{horizon}_shift{horizon}',

        f'snap_CA_target_diff_{horizon}_shift{horizon}',

        f'snap_TX_target_diff_{horizon}_shift{horizon}',

        f'snap_WI_target_diff_{horizon}_shift{horizon}',

    ]

    return feature_columns, linear_feature_columns
%%time



idx_feature = ['id']

categorical_feature = [

    'wday',

    'month',

    'year',

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

cat_features_to_encoding = [

    'wday',

    'month',

    'year',

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

encoded_cat_features = [i + '_target' for i in cat_features_to_encoding]

#train_generated_features = ['day_min', 'day_mean', 'day_max']



label_column = 'sales'



sub_columns = ['id'] + ['F%s' % i for i in range(1, 29)]

submission_validation = pd.DataFrame(columns=sub_columns)

submission_evaluation = pd.DataFrame(columns=sub_columns)

submission_validation_lgb = pd.DataFrame(columns=sub_columns)

submission_evaluation_lgb = pd.DataFrame(columns=sub_columns)

submission_validation_sgd = pd.DataFrame(columns=sub_columns)

submission_evaluation_sgd = pd.DataFrame(columns=sub_columns)

submission_validation_linear = pd.DataFrame(columns=sub_columns)

submission_evaluation_linear = pd.DataFrame(columns=sub_columns)

submission_validation_mean = pd.DataFrame(columns=sub_columns)

submission_evaluation_mean = pd.DataFrame(columns=sub_columns)



random.seed(2)

samples_idx = random.sample(range(train_data.shape[0]), 300)



models = {'linear': 0, 'lgb': 0, 'mean': 0, 'linearSGD': 0}



feature_importance_init =False

feature_importance = None

feature_importance_n = 0



#for iteration, i in enumerate(samples_idx):

for iteration, i in enumerate(range(train_data.shape[0])):  

    if iteration % 1000 == 1:

        print(iteration, train_data.shape[0])

        print('mean', get_rmse(submission_validation_mean))

        print('linear', get_rmse(submission_validation_linear))

        print('sgd', get_rmse(submission_validation_sgd))

        print('lgb', get_rmse(submission_validation_lgb))

        print('pred', get_rmse(submission_validation))

    

    row = train_data.loc[i]

    all_id = row.id

    item_id = row.item_id

    dept_id = row.dept_id

    cat_id = row.cat_id

    store_id = row.store_id

    state_id = row.state_id

    sales = row['d_1':]

    item_df = calendar.join(sales.to_frame('sales'), on='d')

    

    # add prices

    item_df = item_df.join(prices.loc[store_id, item_id], on=['wm_yr_wk'])

    item_df.sales.fillna(0, inplace=True)

    item_df.loc[:,'sales'] = item_df.sales.astype('int64')

    # drop early zeros rows

    item_df = item_df.set_index('d')

    

    first_sale = item_df[item_df.sales!=0].index[0]

    first_sale_int = int(first_sale[2:]) # 'd_1914' -> 1914

    if first_sale_int > 1914 - 90:

        first_sale = 'd_{}'.format(1914  - 90) # garanted 90 days history 

    item_df = item_df.loc[first_sale:,:]

    

    # apply target encoding

    item_df = apply_target_encoder(item_df, cat_features_to_encoding)

    

    feature_columns = []

    linear_feature_columns = []

    # create features

    for horizon in [28,]:

        lag = horizon // 7 

        # add price features

        create_price_features(item_df, horizon=horizon)

        # add categorical features

        create_categorical_features(item_df, horizon=horizon)

        # add sales features

        create_sales_features(item_df, horizon=horizon)

        features, linear_features = get_created_features(horizon=horizon)

        

        feature_columns += features

        linear_feature_columns = linear_features



    feature_columns = feature_columns + encoded_cat_features + categorical_feature

    

    # drop rows with na

    item_df.dropna(inplace=True)

    

    predictions_list = []

        

    try:

        train_df, valid_df, test_df, valid_models_df = get_dataset_simple(item_df[feature_columns + [label_column]], label_column=label_column)

        

        sc_X = StandardScaler()

        sc_y = StandardScaler()

        X_train = sc_X.fit_transform(train_df[linear_feature_columns])

        y_train = sc_y.fit_transform(train_df[label_column].values.reshape(-1,1))



        X_test = sc_X.fit_transform(test_df[linear_feature_columns])



        X_valid = sc_X.fit_transform(valid_df[linear_feature_columns])

        y_valid = sc_y.fit_transform(valid_df[label_column].values.reshape(-1,1))



        model_sgd = SGDRegressor(max_iter=3000)

        model_sgd.fit(X_train, y_train.reshape(-1))

        prediction_sgd = sc_y.inverse_transform(model_sgd.predict(X_test).reshape(-1,1)).reshape(-1).tolist()

        predictions_list.append(['linearSGD', prediction_sgd, mean_squared_error(valid_models_df.sales.values, prediction_sgd[0:28])])

        

        model = LinearRegression()

        model.fit(X_train, y_train.reshape(-1))

        prediction_linear = sc_y.inverse_transform(model.predict(X_test).reshape(-1,1)).reshape(-1).tolist()

        predictions_list.append(['linear', prediction_linear, mean_squared_error(valid_models_df.sales.values, prediction_linear[0:28])])

        

        #lgb

        dtrain = lgb.Dataset(train_df[feature_columns], label=train_df[label_column], categorical_feature=categorical_feature)

        dvalid = lgb.Dataset(valid_df[feature_columns], label=valid_df[label_column], categorical_feature=categorical_feature)



        param = {

            'boosting_type': 'gbdt',

            'objective': 'tweedie',

            #'tweedie_variance_power': 1.1,

            'metric': 'rmse',

            'subsample': 0.5,

            'subsample_freq': 1,

            'learning_rate': 0.03,

            'num_leaves': 64,

            #'max_depth': 7,

            'min_data_in_leaf': min(1024, test_df.shape[0] // 4),

            'feature_fraction': 0.1,

            #'max_bin': 10,

            'boost_from_average': False,

            'verbose': -1,

            'lambda_l1': 0.8,

            #'lambda_l2': 0,

            #'min_gain_to_split': 1.,

            #'min_sum_hessian_in_leaf': 1e-3,

        }

        # https://lightgbm.readthedocs.io/en/latest/index.html

        bst = lgb.train(param, dtrain, valid_sets=[dvalid], num_boost_round = 2400, early_stopping_rounds=500, verbose_eval=False, categorical_feature=categorical_feature)

        if not feature_importance_init:

            feature_importance_init = True

            feature_importance = bst.feature_importance()  

        else:

            feature_importance += bst.feature_importance()

        feature_importance_n += 1



        prediction_lgb = bst.predict(test_df[feature_columns]).tolist()

        predictions_list.append(['lgb', prediction_lgb, mean_squared_error(valid_models_df.sales.values, prediction_lgb[0:28])])

    except Exception as e:

        print(i, all_id, item_df.shape)

        print(e)

        prediction_lgb = test_df[f'sales_mean_rolling_4_wday_shift_{lag}'].values.tolist()

        

    prediction_mean = (test_df[f'sales_mean_rolling_4_wday_shift_{lag}']).values.tolist()

    predictions_list.append(['mean', prediction_mean, mean_squared_error(valid_models_df.sales.values, prediction_mean[0:28])])

    

    

    best_predictions = sorted(predictions_list, key=lambda x: x[2])[0]

    models[best_predictions[0]] += 1



    key = all_id[:-10] + 'validation'

    submission_validation.loc[len(submission_validation)] = [key] + best_predictions[1][0:28]

    submission_evaluation.loc[len(submission_evaluation)] = [all_id] + best_predictions[1][28:56]

    

    submission_validation_lgb.loc[len(submission_validation_lgb)] = [key] + prediction_lgb[0:28]

    submission_evaluation_lgb.loc[len(submission_evaluation_lgb)] = [all_id] + prediction_lgb[28:56]

    

    submission_validation_sgd.loc[len(submission_validation_sgd)] = [key] + prediction_sgd[0:28]

    submission_evaluation_sgd.loc[len(submission_evaluation_sgd)] = [all_id] + prediction_sgd[28:56]

    

    submission_validation_linear.loc[len(submission_validation_linear)] = [key] + prediction_linear[0:28]

    submission_evaluation_linear.loc[len(submission_evaluation_linear)] = [all_id] + prediction_linear[28:56]

    

    submission_validation_mean.loc[len(submission_validation_mean)] = [key] + prediction_mean[0:28]

    submission_evaluation_mean.loc[len(submission_evaluation_mean)] = [all_id] + prediction_mean[28:56]



print('Final')

print('mean', get_rmse(submission_validation_mean))

print('linear', get_rmse(submission_validation_linear))

print('sgd', get_rmse(submission_validation_sgd))

print('lgb', get_rmse(submission_validation_lgb))

print('pred', get_rmse(submission_validation))

print(models)
feature_imp = sorted(list(zip(feature_columns, (feature_importance / feature_importance_n).tolist())), key=lambda x: x[1], reverse=True)

for i, j in feature_imp:

    print(i, j)
'''Final

mean 5.935171130952381

linear 8.818988455178928

sgd 8.815847165302436

lgb 4.544536117878011

pred 4.426746593113699

{'linear': 23, 'lgb': 240, 'mean': 24, 'linearSGD': 13}

CPU times: user 12min 49s, sys: 9.53 s, total: 12min 58s

Wall time: 4min 29s'''

pass
'''Final

mean 5.935171130952381

linear 4.912764838532745

sgd 4.916746732656525

lgb 4.590960128500298

pred 4.201384342558303

{'linear': 60, 'lgb': 137, 'mean': 25, 'linearSGD': 78}

CPU times: user 8min 22s, sys: 7.44 s, total: 8min 30s

Wall time: 3min 56s'''

pass # only categorical_feature
'''Final

mean 5.935171130952381

linear 4.912764838532745

sgd 4.91254679162634

lgb 4.952590789885752

pred 4.577687654284275

{'linear': 71, 'lgb': 147, 'mean': 21, 'linearSGD': 61}

CPU times: user 12min 3s, sys: 9.76 s, total: 12min 13s

Wall time: 4min 52s'''

pass
submission = submission_validation.append(submission_evaluation)

submission.to_csv('/kaggle/working/my_submission.csv', index=False)
submission_mean = submission_validation_mean.append(submission_evaluation_mean)

submission_mean.to_csv('/kaggle/working/my_submission_mean.csv', index=False)
submission_lgb = submission_validation_lgb.append(submission_evaluation_lgb)

submission_lgb.to_csv('/kaggle/working/my_submission_lgb.csv', index=False)
submission_linear = submission_validation_linear.append(submission_evaluation_linear)

submission_linear.to_csv('/kaggle/working/my_submission_linear.csv', index=False)
submission_sgd = submission_validation_sgd.append(submission_evaluation_sgd)

submission_sgd.to_csv('/kaggle/working/my_submission_sgd.csv', index=False)