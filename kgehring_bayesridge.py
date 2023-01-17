# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
from sklearn.linear_model import LinearRegression

from sklearn.linear_model import BayesianRidge

from sklearn.ensemble import IsolationForest



from xgboost import XGBRegressor
pd.set_option('display.max_rows', 500)

pd.set_option('display.max_columns', 500)

pd.set_option('display.width', 150)

pd.options.display.float_format = '{:20,.2f}'.format
data_dict = {}

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        key = filename.partition('.csv')[0]

        data_dict[key] = pd.read_csv(os.path.join(dirname, filename))

data_dict.keys()
sales_train = data_dict['sales_train']
sales_train.head()
def index_cols(df, series_name):

    '''

    This function accepts a pivot table with column names in the format 1, 2, 3, ... and renames each

    column into a tuple such that the first element is the original column name and the second element

    is a suffix passed as an argument ("series_name")    

    '''

    for col in df.columns:

        df = df.rename(columns={col:(col, series_name)})

    

    return df
# Append cnt data together yielding a single dataframe per item-store with the full history of sales for that item-store

# as well as the average sale cnt of the item for each period and the average sale cnt at the store for each time period



shop_item_cnt = pd.pivot_table(sales_train, values='item_cnt_day', index=['shop_id', 'item_id'], columns=['date_block_num'], aggfunc=np.sum)

shop_item_cnt = shop_item_cnt.fillna(0) 

shop_item_cnt = shop_item_cnt.astype('int64')

shop_item_cnt = index_cols(shop_item_cnt, 'cnt')

shop_item_cnt = shop_item_cnt.reset_index()



shop_cnt = pd.pivot_table(sales_train, values='item_cnt_day', index='shop_id', columns=['date_block_num'], aggfunc=np.sum)

shop_cnt = shop_cnt.fillna(0) 

shop_cnt = shop_cnt.astype('int64')

shop_cnt = index_cols(shop_cnt, 'shop_total_cnt')

shop_cnt = shop_cnt.reset_index()



item_cnt = pd.pivot_table(sales_train, values='item_cnt_day', index='item_id', columns=['date_block_num'], aggfunc=np.sum)

item_cnt = item_cnt.fillna(0) 

item_cnt = item_cnt.astype('int64')

item_cnt = index_cols(item_cnt, 'item_total_cnt')

item_cnt = item_cnt.reset_index()
shop_item_cnt = shop_item_cnt.merge(shop_cnt, how='left', left_on=['shop_id'], right_on=['shop_id'])

shop_item_cnt = shop_item_cnt.merge(item_cnt, how='left', left_on=['item_id'], right_on=['item_id'])
def format_train(df, target_month, max_lag=3):

    '''

    Takes the pivot table dataset prepared earlier and grabs the month of interest. The number of preceding months

    defined in the max_lag argument are also selected and included in the results set.

    

    Transfoms the pivot data into one row per store-item combo for a specific month with the preceding n months history.

    '''

    

    df = df.copy()

    for col in df.columns:

        if type(col) is tuple:

            if col[0] >= target_month - max_lag and col[0] <= target_month:

                new_col_name = (target_month-col[0], col[1])

                df = df.rename(columns={col:new_col_name})

            else:

                df = df.drop(columns=col)

    

    return df
def remove_outliers(df, contamination=0.01):

    

    df = df.copy()

    

    clf = IsolationForest(random_state=0, contamination=contamination)

    out = clf.fit_predict(df.drop(columns=['shop_id', 'item_id']))

    

    df['outlier'] = out

    df = df.loc[df['outlier'] != -1]

    df = df.drop(columns=['outlier'])

    

    return df
def rmse(y, yhat):

    '''Simple function to calculte the RMSE between two series of equal lengths'''

    rmse = (((yhat - y)**2).mean())**(1/2)

    return rmse
rmse_linear_total = []

rmse_bays_total = []

rmse_xgb_total = []



for r in range(31, 34):

    

    forecast_month = r

    

    # My forecast model relies on taking the forecast month from prior years and using that data to predict sales for this year. So to predict

    # November 2015 we'd look at the n months leading up to November 2014 and Noveber 2013, build a model off that data, and use it to forecast

    # November 2015.

        

    # Grab labels from this year (test set) and prior years (trianing set)

    yo0y = format_train(shop_item_cnt, forecast_month)

    yo1y = format_train(shop_item_cnt, forecast_month-12)

    yo2y = format_train(shop_item_cnt, forecast_month-24)

    

    # Consolidate prior year's data in into asingle dataset

    consolidated_hist = pd.concat([yo1y, yo2y])

    consolidated_hist = remove_outliers(consolidated_hist, 0.01)

    

    # Select training and test sets and labels

    X_train = consolidated_hist.drop(columns=[(0,'cnt'),(0,'shop_total_cnt'),(0,'item_total_cnt')])

    y_train = consolidated_hist[(0,'cnt')].to_numpy()

    X_test = yo0y.drop(columns=[(0,'cnt'),(0,'shop_total_cnt'),(0,'item_total_cnt')])

    y_test = yo0y[(0,'cnt')].to_numpy()

    

    #Model and forecast

    m_linear = LinearRegression()

    m_linear.fit(X_train, y_train)

    l_y_hat = m_linear.predict(X_test)

    

    m_bayes = BayesianRidge()

    m_bayes.fit(X_train, y_train)

    b_y_hat = m_bayes.predict(X_test)

    

    m_xgb = XGBRegressor()

    m_xgb.fit(X_train, y_train)

    x_y_hat = m_xgb.predict(X_test)

    

    rmse_linear = rmse(y_test, l_y_hat)

    rmse_bayes = rmse(y_test, b_y_hat)

    rmse_xgboost = rmse(y_test, x_y_hat)

    

    print('Forecast Month: {0}'.format(forecast_month))

    print('LinearRegression RMSE: {0}'.format(rmse_linear))

    print('BayesianRidge RMSE: {0}'.format(rmse_bayes))

    print('XGBoost RMSE: {0}'.format(rmse_xgboost))

    

    rmse_linear_total.append(rmse_linear)

    rmse_bays_total.append(rmse_bayes)

    rmse_xgb_total.append(rmse_xgboost)



# Print overall evaluations

print('Mean')

print('LinearRegression RMSE: {0}'.format(np.mean(rmse_linear_total)))

print('BayesianRidge RMSE: {0}'.format(np.mean(rmse_bays_total)))

print('XGBoost RMSE: {0}'.format(np.mean(rmse_xgb_total)))
forecast_month = 34



yo0y = format_train(shop_item_cnt, forecast_month).reset_index()

yo1y = format_train(shop_item_cnt, forecast_month-12).reset_index()

yo2y = format_train(shop_item_cnt, forecast_month-24).reset_index()





consolidated_hist = pd.concat([yo1y, yo2y])

consolidated_hist = remove_outliers(consolidated_hist)



X_train = consolidated_hist.drop(columns=[(0,'cnt'),(0,'shop_total_cnt'),(0,'item_total_cnt')])

y_train = consolidated_hist[(0,'cnt')].to_numpy()

X_test = yo0y

m = BayesianRidge()





m.fit(X_train, y_train)



y_hat = m.predict(X_test)
i = X_test.copy()

i['item_cnt_month'] = y_hat



test = data_dict['test']



test = test.merge(i, how='left', left_on=['shop_id', 'item_id'], right_on=['shop_id', 'item_id'])



test.loc[test['item_cnt_month'].isna() == True]



test = test[['ID', 'item_cnt_month']].fillna(0)



test.to_csv('output.csv', index=False)