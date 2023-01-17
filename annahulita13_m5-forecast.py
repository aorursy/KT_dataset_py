# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.preprocessing import LabelEncoder

from sklearn import metrics

import lightgbm as lgb

from sklearn.linear_model import Ridge

import gc



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# d_1 - d_1913 - train

# d_1914 - d_1941 - validation

# d_1942 - d_1969 - evaluation



# validation_start = 1914

# evaluation_start = 1942



# train_start = 1500

# prediction_start = 1914
# # avg prediction

# def get_train(sales, train_start, train_end):

#     # train_start - included

#     # train_end - excluded

#     train_range = range(train_start,train_end)

#     d_train  = ['d_' + str(i) for i in train_range]

#     sales_columns = ['id','item_id','dept_id','cat_id','store_id','state_id']

#     train = sales[sales_columns + d_train]



#     train = train.rename(dict(zip(d_train, train_range)), axis=1)

#     train['id'] = train['id'].str.replace('_evaluation','')

#     train = train.melt(id_vars=sales_columns, value_vars=train_range, var_name='d', value_name='demand')

#     return train



# def predict_avg(train, avg_n):

#     # predict avg

#     train_end = train['d'].max()

#     avg_data = train[train['d']>(train_end-avg_n)]

#     avg_data = avg_data.groupby(['id'])['demand'].mean()

#     avg_data = pd.DataFrame(avg_data)





#     f_columns = ['F' + str(i+1) for i in range(0,28)]

#     for column in f_columns:

#         avg_data[column] = avg_data['demand']



#     avg_data = avg_data[f_columns]

#     avg_data.reset_index(inplace=True)

#     return avg_data



# def submit_avg():

#     train = get_train(sales, 1800, validation_start)

#     avg_data_val = predict_avg(train, 28)

#     avg_data_val['id'] = avg_data_val['id'] + '_validation'

    

#     train = get_train(sales, 1800, evaluation_start)

#     avg_data_eval = predict_avg(train, 28)

#     avg_data_eval['id'] = avg_data_eval['id'] + '_evaluation'

    

#     avg_data = pd.concat([avg_data_val,avg_data_eval])

    

#     submission = submission[['id']].merge(avg_data, how='left')

#     submission.to_csv("submission28.csv", index=False)
def reshape_sales(sales, train_start, train_end):

    # train_start - included

    # train_end - excluded

    train_range = range(train_start,train_end)

    d_train  = ['d_' + str(i) for i in train_range]

    sales_columns = ['id','item_id','dept_id','cat_id','store_id','state_id']

    sales = sales[sales_columns + d_train]

    

    sales = sales.reindex(columns=sales.columns.tolist() + ["d_" + str(train_end + i) for i in range(28)])



    sales['id'] = sales['id'].str.replace('_evaluation','')

    sales = sales.melt(id_vars=sales_columns, var_name='d', value_name='demand')

    sales['d'] = sales['d'].str[2:].astype("int16")

    

    calendar['event'] = (~calendar['event_name_1'].isna()|~calendar['event_name_2'].isna()).astype(int)

    sales = sales.merge(calendar[['d','wday','event']], how='left', on=['d'])

    return sales
def prepare_sales(sales):

    le = LabelEncoder()

    sales['wday'] = le.fit_transform(sales['wday'])

    

    le = LabelEncoder()

    sales['cat_id'] = le.fit_transform(sales['cat_id'])



    le = LabelEncoder()

    sales['dept_id'] = le.fit_transform(sales['dept_id'])



    sales['lag28'] = sales.groupby(['id'])['demand'].transform(lambda x: x.shift(28))

    sales['lag30'] = sales.groupby(['id'])['demand'].transform(lambda x: x.shift(30))

    

    sales['rolling_mean_7'] = sales.groupby(['id'])['demand'].transform(lambda x: x.shift(28).rolling(7).mean())

    sales['rolling_mean_3'] = sales.groupby(['id'])['demand'].transform(lambda x: x.shift(28).rolling(3).mean())

    sales['rolling_mean_30'] = sales.groupby(['id'])['demand'].transform(lambda x: x.shift(28).rolling(30).mean())

    

    sales['rolling_std_7'] = sales.groupby(['id'])['demand'].transform(lambda x: x.shift(28).rolling(7).std())

    

    #sales = sales[~sales['rolling_mean_7'].isna()]

    sales.sort_values(['id','d'], inplace=True)

    sales.fillna(method = 'bfill', inplace=True)



    return sales
def train_predict_lgb(sales, prediction_start):

    d_min = sales['d'].min()

    d_max = prediction_start-1

    val_split_d = d_min+int((d_max-d_min)*0.9)

    

    test = sales[sales['d']>=prediction_start]

    val = sales[(sales['d']<prediction_start)&(sales['d']>=val_split_d)]

    train = sales[sales['d']<val_split_d][~sales['rolling_mean_7'].isna()]

        

    train_set = lgb.Dataset(train[features], label=train[label], feature_name=features, categorical_feature=cat_features)

    val_set = lgb.Dataset(val[features], label=val[label], feature_name=features, categorical_feature=cat_features)

    

    params = {

        'boosting_type': 'gbdt',

        'metric': 'rmse',

        'objective': 'regression',

        'n_jobs': -1,

        'seed': 236,

        'learning_rate': 0.1,

        'bagging_fraction': 0.75,

        'bagging_freq': 10, 

        'colsample_bytree': 0.75}

    

    model = lgb.train(params, train_set, num_boost_round = 1000, early_stopping_rounds = 50, valid_sets = [train_set, val_set], verbose_eval = 100)



    val_pred = model.predict(val[features])

    val_score = np.sqrt(metrics.mean_squared_error(val_pred, val[label]))

    print('val rmse score is {}'.format(val_score))



    test_pred = model.predict(test[features])

    test['demand']=test_pred

    

    return test
def train_predict_ridge(sales, prediction_start):

    test = sales[sales['d']>=prediction_start]

    train = sales[sales['d']<prediction_start]



    model = Ridge(alpha=1.0)

    

    model.fit(train[features], train[label])

    test_pred = model.predict(test[features])

    test['demand'] = test_pred

    

    return test
def submit(sales, train_start, train_predict_func, mode='validation'):

    if mode == 'validation':

        prediction_start = 1914

    elif mode == 'evaluation':

        prediction_start = 1942

    

    print('reshaping sales ...')

    sales = reshape_sales(sales, train_start, prediction_start)

    print('preparing sales ...')

    sales = prepare_sales(sales)

    print('training and predicting ...')

    test = train_predict_func(sales, prediction_start)

    

    prediction = pd.pivot(test[['id','d','demand']], index = 'id', columns = 'd', values = 'demand')

    prediction.columns = ['F'+str(i+1) for i in range(0,28)]

    prediction.reset_index(inplace=True)

    prediction['id'] = prediction['id']+'_'+mode

    return prediction
def reshape_validation(sales):

    # get validation ground truth for comparison

    d_validation  = ['d_' + str(i) for i in range(1914, 1914+28)]

    sales_validation = sales[['id']+d_validation]

    sales_validation['id'] = sales_validation['id'].str.replace('_evaluation','')

    sales_validation = sales_validation.melt(id_vars=['id'], var_name='d', value_name='val_demand')

    sales_validation['d'] = sales_validation['d'].str[2:].astype("int16")

    return sales_validation



def get_rmse_score(sales_validation, test):

    sales_validation = sales_validation[['id','d','val_demand']].merge(test[['id','d','demand']], on=['id','d'],how='left')

    rmse = np.sqrt(metrics.mean_squared_error(sales_validation['val_demand'], sales_validation['demand']))

    return rmse
features = ['wday', 'lag28', 'rolling_mean_7', 'rolling_std_7', 'lag30', 'rolling_mean_3', 'rolling_mean_30', 'event','cat_id','dept_id']

cat_features = ['wday', 'event','cat_id','dept_id'] 

label = 'demand'

# dev mode

# submission = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/sample_submission.csv')

# calendar = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/calendar.csv')

# calendar['d'] = calendar['d'].str[2:].astype("int16")



# sales = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/sales_train_evaluation.csv')

# sales_validation = reshape_validation(sales)



# train_start, prediction_start = 1000, 1914



# print('reshaping sales ...')

# sales = reshape_sales(sales, train_start, prediction_start)

# print('preparing sales ...')

# sales = prepare_sales(sales)



# test = train_predict_lgb(sales, prediction_start)

# rmse = get_rmse_score(sales_validation, test)

# rmse
# 2.2497847944992713
submission = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/sample_submission.csv')

calendar = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/calendar.csv')

calendar['d'] = calendar['d'].str[2:].astype("int16")



sales = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/sales_train_evaluation.csv')

prediction_val = submit(sales, 1000, train_predict_lgb, mode='validation')



sales = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/sales_train_evaluation.csv')

prediction_eval = submit(sales, 1000, train_predict_lgb, mode='evaluation')



prediction_data = pd.concat([prediction_val,prediction_eval])

submission = submission[['id']].merge(prediction_data, how='left')

submission.to_csv("submission_lgb.csv", index=False)