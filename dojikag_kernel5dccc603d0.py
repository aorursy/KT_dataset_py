# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
from datetime import datetime as dt

from sklearn.linear_model import LinearRegression

from sklearn.ensemble import GradientBoostingRegressor

# from sklearn.grid_search import GridSearchCV

from sklearn.model_selection import GridSearchCV

from sklearn.preprocessing import StandardScaler

#import statsmodels.tsa as sm



if __name__ == '__main__':

    df = pd.read_csv('../input/train.csv')

    print('read train file')

    df['date'] = df['date'].apply(lambda x:dt.strptime(str(x),"%Y-%m-%d"))

    df_test = pd.read_csv('../input/test.csv')

    df_test['date'] = df_test['date'].apply(lambda x:dt.strptime(str(x),"%Y-%m-%d"))

    eval_term = len(df_test['date'].unique().tolist())

    

    df_time = pd.concat([df[['date']], df_test[['date']]])

    df_time = df_time.groupby('date').sum()

    df_time.reset_index(inplace=True)

    

    # item_all

    df_item = df[['item', 'date', 'sales']].groupby(['item','date']).sum()

    df_item.reset_index(inplace=True)

    

    # store_all

    df_store = df[['store', 'date', 'sales']].groupby(['store','date']).sum()

    df_store.reset_index(inplace=True)

    

    # calendar

    df_day = df_time.copy()

    df_day['year'] = df_day['date'].apply(lambda x:x.year)

    df_day['month'] = df_day['date'].apply(lambda x:x.month)

    df_day = df_day.groupby(['date']).first()

    df_day.reset_index(inplace=True)

    df_wday = df_day.copy()

    df_wday['wday'] = df_wday['date'].apply(lambda x : x.isoweekday())

    df_wday = pd.get_dummies(df_wday[['wday']], columns=['wday']).astype(int)

    df_month = df_day.copy()

    df_month = pd.get_dummies(df_month[['month']], columns=['month']).astype(int)

    df_day = pd.concat([df_day, df_wday], axis=1)

    df_day = pd.concat([df_day, df_month], axis=1)

    ls_day = df_day.columns.tolist()

    ls_day.remove('date')

    print('make calendar data')

    

    # calendar_train/test

    train_expv = df_day[ls_day].iloc[:-eval_term].values

    test_expv = df_day[ls_day].values

    

    # prediciton(store)

    ls_store = df_store['store'].unique().tolist()

    df_st_lr = pd.DataFrame()

    for i in ls_store:

        df_st_tmp = df_store[df_store['store'] == i].copy()

        train_objv = df_st_tmp['sales'].values

        lr = LinearRegression(fit_intercept=False)

        res_lr = lr.fit(train_expv, train_objv)

        df_result = df_day[['date']].copy()

        df_result['store'] = i

        df_result['store_lr'] = res_lr.predict(test_expv)

        if(len(df_st_lr)==0):

            df_st_lr = df_result.copy()

        else:

            df_st_lr = df_st_lr.append(df_result)

    print('make predict(store) data')

    

    # predict(item)

    ls_item = df_item['item'].unique().tolist()

    df_it_lr = pd.DataFrame()

    for i in ls_item:

        df_it_tmp = df_item[df_item['item'] == i].copy()

        train_objv = df_it_tmp['sales'].values

        lr = LinearRegression(fit_intercept=False)

        res_lr = lr.fit(train_expv, train_objv)

        df_result = df_day[['date']].copy()

        df_result['item'] = i

        df_result['item_lr'] = res_lr.predict(test_expv)

        if(len(df_it_lr)==0):

            df_it_lr = df_result.copy()

        else:

            df_it_lr = df_it_lr.append(df_result)

    print('make predict(item) data')

    

    df['target'] = df['store'].astype(str) + '_' + df['item'].astype(str)

    df_test['target'] = df_test['store'].astype(str) + '_' + df_test['item'].astype(str)

    ls_target = df_test['target'].unique().tolist()

    #ls_expv = ['store_lr','item_lr','exps_sim','exps_all','holt']

    ls_expv = ['store_lr','item_lr']

    print('finish prepareing process')

    

    df_out = pd.DataFrame()

    for i in ls_target:

        print('target(store_item):',i)

        df_tmp = df[df['target'] == i ]

        df_tmp = pd.merge(df_time, df_tmp, on=['date'], how='left')

        df_tmp['item'] = df_tmp['item'].iloc[0]

        df_tmp['store'] = df_tmp['store'].iloc[0]

        df_tmp['target'] = df_tmp['target'].iloc[0]

        df_tmp = pd.merge(df_tmp, df_st_lr, on=['date', 'store'], how='left')

        df_tmp = pd.merge(df_tmp, df_it_lr, on=['date', 'item'], how='left')

        

        #model1 = sm.holtwinters.ExponentialSmoothing(df_tmp['sales'].iloc[:-eval_term], trend='add').fit()

        #model2 = sm.holtwinters.ExponentialSmoothing(df_tmp['sales'].iloc[:-eval_term], seasonal_periods=7, trend='add', seasonal='add').fit()

        #model3 = sm.holtwinters.Holt(df_tmp['sales'].iloc[:-eval_term]).fit()

        #df_tmp['exps_sim'] = model1.predict(0,len(df_tmp))

        #df_tmp['exps_all'] = model2.predict(0,len(df_tmp))

        #df_tmp['holt'] = model3.predict(0,len(df_tmp))

        

        train_expv = df_tmp[ls_expv].iloc[:-eval_term].values

        test_expv = df_tmp[ls_expv].values

        train_objv = df_tmp['sales'].iloc[:-eval_term].values

        bool_scaler = True

        if (bool_scaler):

            scaler = StandardScaler()

            train_expv = scaler.fit_transform(train_expv)

            test_expv = scaler.fit_transform(test_expv)

            train_objv = scaler.fit_transform(train_objv.reshape(-1,1)).ravel()

        

        # GBreg

        tuned_parameters =[{'n_estimators':[10,20,50],

                           'max_features':['auto'],

                           'random_state':[0],

                           'min_samples_split':[5,10],

                           'max_depth':[5,10]

                           }]

        clf = GridSearchCV(GradientBoostingRegressor(), tuned_parameters, cv=5, scoring='neg_mean_squared_error')

        clf.fit(train_expv, train_objv)

        res_rf = clf.best_estimator_

        predict = res_rf.predict(test_expv)

        

        if (bool_scaler):

            predict = scaler.inverse_transform(predict)

            

        df_tmp['predict'] = predict

        df_tmp = df_tmp[['date','store','item','sales','predict']].iloc[-eval_term:] # -462:

        

        if(len(df_out) == 0):

            df_out = df_tmp.copy()

        else:

            df_out = df_out.append(df_tmp)

            #break

    df_test = pd.read_csv('../input/test.csv', dtype={'store':int,'item':int})

    df_test['date'] = df_test['date'].apply(lambda x:dt.strptime(str(x),"%Y-%m-%d"))

    df_test = pd.merge(df_test, df_out[['date','store','item','predict']], on=['date','store','item'], how='left')

    df_test.rename(columns={'predict':'sales'}, inplace=True)

    df_test = df_test[['id','sales']]

    print(df_test.head(10))

    df_test.to_csv('output.csv',index=False)