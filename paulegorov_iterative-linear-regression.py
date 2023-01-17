import pandas as pd

import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_absolute_error

from datetime import datetime, timedelta, date



import numpy as np



def smape(A, F):

    return 100/len(A) * np.sum(2 * np.abs(F - A) / (np.abs(A) + np.abs(F)))

df_train = pd.read_csv('/kaggle/input/demand-forecasting-kernels-only/train.csv')

df_test = pd.read_csv('/kaggle/input/demand-forecasting-kernels-only/test.csv')



df_train.info()

df_test.info()
df = pd.concat([df_train, df_test], sort = False)

df['date'] = pd.to_datetime(df['date'])

df = df.set_index('date')

df
store_list = df['store'].value_counts().index.values

item_list = df['item'].value_counts().index.values



store_cnt = len(store_list)

item_cnt = len(item_list)

print('Total stores', store_cnt)

print('Total items', item_cnt)
plt.figure(figsize = (10,5))

for store_id in range(store_cnt):

    plt.plot(df.query("store == @store_id and item == 1")['sales'])
def make_features(data, min_lag, max_lag, rolling_mean_size):

    data['year'] = data.index.year

    data['month'] = data.index.month

    data['day'] = data.index.day

    data['dayofweek'] = data.index.dayofweek

    data['lag_365'] = data['sales'].shift(365)

    

    for lag in range(min_lag, max_lag + 1):

        data['lag_{}'.format(lag)] = data['sales'].shift(lag)

        

    data['rolling_mean'] = data['sales'].shift().rolling(rolling_mean_size).mean()
import warnings

warnings.filterwarnings("ignore")



submission_df = pd.DataFrame(columns = ['id', 'sales'])



for s in store_list:

    for i in item_list:



        #dataframe for current store/item pair

        df_item_store = df.query("store == @s and item == @i")



        #creating features 

        make_features(df_item_store, 1, 28, 30)



        #training LR on non-empty data

        df_train = df_item_store.query("date >= '2014-01-01' and date <='2017-12-31'").drop('id', axis = 1)

        X = df_train.drop('sales', axis = 1)

        y = df_train['sales']

        train_X, valid_X, train_y, valid_y = train_test_split(X,y, shuffle=False, test_size=0.2)



        model = LinearRegression()

        model.fit(train_X, train_y)

        pred_train = model.predict(train_X)

        pred_valid = model.predict(valid_X)



        print("SMAPE for store %s and item %s: %.2f/%.2f" % (s, i, smape(train_y, pred_train), smape(valid_y, pred_valid)))



        #for all dates of test set iterativly create features and predict sales row by row

        start_date = date(2018, 1, 1)

        end_date = date(2018, 3, 31)

        while start_date <= end_date:

            #calculating features for current date in test set

            for lag in range(1, 28 + 1):

                df_item_store.loc[start_date, 'lag_{}'.format(lag)] = df_item_store.loc[start_date-timedelta(days=lag), 'sales'] 

            df_item_store.loc[start_date, 'rolling_mean'] = df_item_store['sales'].shift().rolling(30).mean()[start_date]



            #predict current price

            df_item_store.loc[start_date, 'sales'] = model.predict(df_item_store[start_date:start_date].drop(['id', 'sales'], axis = 1))[0]



            #next date

            start_date += timedelta(days=1)



        #append submission results for current store/item pair

        submission_df = submission_df.append(df_item_store['2018-01-01':][['id', 'sales']])

print('Prediction finished!')

submission_df['id'] = submission_df['id'].astype('int32')

submission_df

#saving submission file

submission_df.to_csv('submission.csv', index = False)