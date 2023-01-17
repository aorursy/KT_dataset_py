# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import lightgbm as lgb



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
sell_prices = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/sell_prices.csv')

sample_submission = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/sample_submission.csv')

calendar = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/calendar.csv')

sales_train_validation = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/sales_train_evaluation.csv')
sales_train_validation
#Do one dataset at a time, and loop thru. Only use the calendar
sales_data = pd.DataFrame(sales_train_validation).reset_index()

for i in range(1942, 1972):

    sales_data['d_' + str(i)] = 0

sell_prices = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/sell_prices.csv')

sell_prices['id'] = sell_prices['item_id'] + '_' + sell_prices['store_id']

sell_prices = sell_prices.pivot(index='id', columns = 'wm_yr_wk', values = 'sell_price').reset_index()

sell_prices = sell_prices.fillna(method='bfill', axis=1)

test = sales_data['item_id'] + '_' + sales_data['store_id']

sell_prices = sell_prices.set_index('id')

sell_prices = sell_prices.reindex(test)

sell_prices = sell_prices.reset_index()
labels = calendar[['d', 'event_name_1', 'event_name_2', 'snap_CA', 'snap_TX', 'snap_WI', 'wday', 'month', 'year', 'wm_yr_wk']].reset_index()

labels = labels.fillna(0)

#labels['wm_yr_wk'] = int(labels['wm_yr_wk'])

labels['event_name_1'] = labels['event_name_1'].astype('str')

labels['event_name_2'] = labels['event_name_2'].astype('str')
from sklearn.preprocessing import LabelEncoder

label_encoder=LabelEncoder()

label_encoder = label_encoder.fit(labels['event_name_1'])

label_encoded_event1 = label_encoder.transform(labels['event_name_1'])

label_encoder = label_encoder.fit(labels['event_name_2'])

label_encoded_event2 = label_encoder.transform(labels['event_name_2'])

labels['event_name_1_encode'] = label_encoded_event1

labels['event_name_2_encode'] = label_encoded_event2
#lgbm Model

params = {



#         'boosting_type': 'gbdt',

        'metric': 'rmse',

        'objective': 'poisson',

        'n_jobs': -1,

        'seed': 20,

        'learning_rate': 0.1,

        'alpha': 0.1,

        'lambda': 0.1,

        'bagging_fraction': 0.66,

        'bagging_freq': 2, 

        'colsample_bytree': 0.77}





results = []

ss = pd.DataFrame(sample_submission)
ss
import time 

from tqdm import tqdm
for i in tqdm(range(0, len(sales_data))):

    temp = pd.DataFrame(sales_data.loc[i][7:]).reset_index()

    temp = temp.rename(columns = {"index": "d", i: "sales"})

    sales_all = labels.merge(temp, on=["d"])

    sales_all = sales_all.merge(sell_prices.reset_index().loc[i], on=['wm_yr_wk'])



    sales_all['lag_28'] = sales_all['sales'].shift(28)

    

    sales_all['rolling_std_t7'] = sales_all[i].transform(lambda x: x.rolling(7).std())

    sales_all['rolling_std_t30'] = sales_all[i].transform(lambda x: x.rolling(30).std())

    

    sales_all['rolling_mean_7'] = sales_all['sales'].transform(lambda x: x.shift(28).rolling(7).mean())

    sales_all['rolling_mean_30'] = sales_all['sales'].transform(lambda x: x.shift(28).rolling(30).mean())

    sales_all['rolling_mean_60'] = sales_all['sales'].transform(lambda x: x.shift(28).rolling(60).mean())

    

    sales_all['lag_price_t1'] = sales_all[i].transform(lambda x: x.shift(1))   

    sales_all['price_change_t1'] = (sales_all['lag_price_t1'] - sales_all[i]) / sales_all['lag_price_t1']

    

    sales_all.drop(['lag_price_t1'], inplace = True, axis = 1)

    

    features = ['event_name_1_encode', 'event_name_2_encode', 'snap_' + sales_data['state_id'][i]

                , 'wday','month', 'year'

                , 'lag_28'

                , 'rolling_std_t7', 'rolling_std_t30'

                , 'rolling_mean_7', 'rolling_mean_30', 'rolling_mean_60'

                , 'price_change_t1', i]

    features_sales = ['event_name_1_encode', 'event_name_2_encode', 'snap_' + sales_data['state_id'][i]

                , 'wday','month', 'year'

                , 'lag_28'

                , 'rolling_std_t7', 'rolling_std_t30'

                , 'rolling_mean_7', 'rolling_mean_30', 'rolling_mean_60'

                , 'price_change_t1',i, 'sales']

    

    sales_all[features_sales] = sales_all[features_sales].applymap(float)



    X_train = lgb.Dataset(sales_all[features][0:1940]

                      , label = sales_all[['sales']][0:1940])

    clf = lgb.train(params, X_train)

    y_pred = clf.predict(sales_all[features][1941:])

   

    ss.loc[i+30490, 1:] = y_pred
ss.to_csv("submission.csv", index=False)
