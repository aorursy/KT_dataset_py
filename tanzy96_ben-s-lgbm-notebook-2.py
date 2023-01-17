import lightgbm as lgb

import seaborn as sns

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression

import statsmodels.api as sm

from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import mean_squared_error, r2_score

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import MinMaxScaler

from scipy.stats import hmean,skew,kurtosis

import math 

import IPython.display as ipd

dtype_dict={"id":np.uint32,

            "store_nbr":np.uint8,

            "item_nbr":np.uint32,

            "unit_sales":np.float16}

item_dtype_dict={"item_nbr":np.uint32,

            "family":np.object,

            "class":np.uint8,

            "perishable":np.uint8}

store_dtype_dict={"store_nbr":np.uint8,

            "store_nbr":np.uint8,

            "city":np.object,

            "state":np.object,

                 "type":np.object,

                 "cluster":np.uint8}
import os

os.listdir("../input")
items = pd.read_csv("../input/items.csv")

items = items.iloc[:,[0,3]]

items.head()
train = pd.read_csv("../input/train.csv",dtype=dtype_dict, converters={'unit_sales': lambda u: np.log1p(float(u)) if float(u) > 0 else 0})

train['Year'] = pd.DatetimeIndex(train['date']).year

train_2017=train.loc[train['Year'] ==2017]

del train 

import gc

gc.collect()
train_promo_2017 = train_2017.set_index(["store_nbr", "item_nbr", "date"])[["onpromotion"]].unstack().fillna(0)

train_sales_2017 = train_2017.set_index(["store_nbr", "item_nbr", "date"])[["unit_sales"]].unstack().fillna(0)

train_sales_2017.columns = train_sales_2017.columns.get_level_values(1)
test = pd.read_csv("../input/test.csv")

test['date'] = pd.to_datetime(test['date'])

test_promo_2017 = test.set_index(["store_nbr", "item_nbr", "date"])[["onpromotion"]].unstack().fillna(0)

test_promo_2017 = test_promo_2017.reindex(train_promo_2017.index).fillna(0)

test_train_promo_2017 = pd.concat([train_promo_2017,test_promo_2017],axis=1)

test_train_promo_2017.columns = test_train_promo_2017.columns.get_level_values(1)
items = pd.read_csv("../input/items.csv")

items = items.iloc[:,[0,2]]

train_class_2017=pd.merge(train_2017,items,how = 'left',on='item_nbr')

train_class_promo_2017 = train_class_2017.set_index(["store_nbr", "item_nbr","class","date"])[["onpromotion"]].unstack().fillna(0)

test = pd.read_csv("../input/test.csv")

test['date'] = pd.to_datetime(test['date'])

test_class_2017 = pd.merge(test,items,how = 'left',on='item_nbr')

test_class_promo_2017 = test_class_2017.set_index(["store_nbr", "item_nbr","class", "date"])[["onpromotion"]].unstack().fillna(0)

test_class_promo_2017 = test_class_promo_2017.reindex(train_class_promo_2017.index).fillna(0)

test_train_class_promo_2017 = pd.concat([train_class_promo_2017,test_class_promo_2017],axis=1)

test_train_class_promo_2017.columns = test_train_class_promo_2017.columns.get_level_values(1)

train_class_sales_2017 = train_class_2017.set_index(["store_nbr", "item_nbr","class","date"])[["unit_sales"]].unstack().fillna(0)

train_class_sales_2017.columns = train_class_sales_2017.columns.get_level_values(1)



def features_dataset_val(dataframe,val_date):

    df_val = []

    y= []

    y = pd.DataFrame(y)

    df_val = pd.DataFrame(df_val)

    month = int(val_date.strftime('%m'))

    week = val_date.isocalendar()[1]

    for i in [10, 20, 30,40,50,60,120]:

        df_val['mean_for_-{}'.format(i)] = dataframe[pd.date_range(start =(val_date-timedelta(days=i)), periods=i, freq = 'D')].mean(axis=1)

        df_val['std_for_-{}'.format(i)] = dataframe[pd.date_range(start =(val_date-timedelta(days=i)), periods=i,freq = 'D')].std(axis=1)

        df_val['var_for_-{}'.format(i)] = dataframe[pd.date_range(start =(val_date-timedelta(days=i)), periods=i, freq = 'D')].var(axis=1)

        df_val['median_for_-{}'.format(i)] = dataframe[pd.date_range(start =(val_date-timedelta(days=i)),periods=i,freq = 'D')].median(axis=1)

        df_val['zero_sales_for_-{}'.format(i)] = dataframe[pd.date_range(start =(val_date-timedelta(days=i)), periods=i, freq = 'D')].apply(lambda x : list(x).count(0),axis=1)

        df_val['kurtosis_for_-{}'.format(i)] = dataframe[pd.date_range(start =(val_date-timedelta(days=i)), periods=i, freq = 'D')].apply(kurtosis, axis=1)

        df_val['hmean_for_-{}'.format(i)] = (dataframe[pd.date_range(start =(val_date-timedelta(days=i)), periods=i, freq = 'D')]+0.0001).apply(hmean, axis=1)

        df_val['skew_for_-{}'.format(i)] = dataframe[pd.date_range(start =(val_date-timedelta(days=i)), periods=i, freq = 'D')].apply(skew, axis=1)

        df_val['mean_for_class_-{}'.format(i)] = train_class_sales_2017[pd.date_range(start =(val_date-timedelta(days=i)), periods=i, freq = 'D')].mean(axis=1).values

        df_val['zero_sales_for_class_-{}'.format(i)] = train_class_sales_2017[pd.date_range(start =(val_date-timedelta(days=i)), periods=i, freq = 'D')].apply(lambda x : list(x).count(0),axis=1).values



    df_val['promosum{}'.format(16)] = test_train_promo_2017[pd.date_range(start =(val_date-timedelta(days=16)), periods=16, freq = 'D')].sum(axis=1)

    df_val['promosum{}'.format(30)] = test_train_promo_2017[pd.date_range(start =(val_date-timedelta(days=30)), periods=30, freq = 'D')].sum(axis=1)

    df_val['promosum{}'.format(60)] = test_train_promo_2017[pd.date_range(start =(val_date-timedelta(days=60)), periods=60, freq = 'D')].sum(axis=1)

    df_val['promosum_class{}'.format(16)] = test_train_class_promo_2017[pd.date_range(start =(val_date-timedelta(days=16)), periods=16, freq = 'D')].sum(axis=1).values

    df_val['promosum_class{}'.format(30)] = test_train_class_promo_2017[pd.date_range(start =(val_date-timedelta(days=30)), periods=30, freq = 'D')].sum(axis=1).values

    df_val['promosum_class{}'.format(60)] = test_train_class_promo_2017[pd.date_range(start =(val_date-timedelta(days=60)), periods=60, freq = 'D')].sum(axis=1).values

    

    for i in range(7): #one month worth of day of week values

        df_val['mean_for_dow_4_{}'.format(i)] = dataframe[pd.date_range(start =(val_date - timedelta(days=28-i)),periods=4,freq='7D')].mean(axis=1)

        df_val['mean_for_dow_8_{}'.format(i)] = dataframe[pd.date_range(start =(val_date - timedelta(days=56-i)),periods=8,freq='7D')].mean(axis=1)

        df_val['mean_for_dow_16_{}'.format(i)] = dataframe[pd.date_range(start =(val_date - timedelta(days=112-i)),periods=16,freq='7D')].mean(axis=1)

        df_val['mean_for_dow_20_{}'.format(i)] = dataframe[pd.date_range(start =(val_date - timedelta(days=140-i)),periods=20,freq='7D')].mean(axis=1)

        df_val['zero_sales_for_dow_4_{}'.format(i)] = dataframe[pd.date_range(start =(val_date - timedelta(days=28-i)),periods=4,freq='7D')].apply(lambda x : list(x).count(0),axis=1)

        df_val['zero_sales_for_dow_8_{}'.format(i)] = dataframe[pd.date_range(start =(val_date - timedelta(days=56-i)),periods=8,freq='7D')].apply(lambda x : list(x).count(0),axis=1)

        df_val['zero_sales_for_dow_16_{}'.format(i)] = dataframe[pd.date_range(start =(val_date - timedelta(days=112-i)),periods=16,freq='7D')].apply(lambda x : list(x).count(0),axis=1)

        df_val['zero_sales_for_dow_20_{}'.format(i)] = dataframe[pd.date_range(start =(val_date - timedelta(days=140-i)),periods=20,freq='7D')].apply(lambda x : list(x).count(0),axis=1)



    for i in range(16):

        y['day{}'.format(i)] = dataframe[(val_date+timedelta(days=i))]

        df_val['promo_{}'.format(i)] = test_train_promo_2017[val_date+timedelta(days=i)].values

    return df_val,y


def features_dataset_test(dataframe,test_date):

    # find mean,median,std,var of data for the specific store and item number that is a month before

    df_test = []

    df_test = pd.DataFrame(df_test)

    month = int(test_date.strftime('%m'))

    week = test_date.isocalendar()[1]

    for i in [10, 20, 30,40,50,60,120]:

        df_test['mean_for_-{}'.format(i)] = dataframe[pd.date_range(start =(test_date-timedelta(days=i)), periods=i, freq = 'D')].mean(axis=1)

        df_test['std_for_-{}'.format(i)] = dataframe[pd.date_range(start =(test_date-timedelta(days=i)), periods=i, freq = 'D')].std(axis=1)

        df_test['var_for_-{}'.format(i)] = dataframe[pd.date_range(start =(test_date-timedelta(days=i)), periods=i, freq = 'D')].var(axis=1)

        df_test['median_for_-{}'.format(i)] = dataframe[pd.date_range(start =(test_date-timedelta(days=i)), periods=i, freq = 'D')].median(axis=1)

        df_test['zero_sales_for_-{}'.format(i)] = dataframe[pd.date_range(start =(test_date-timedelta(days=i)), periods=i, freq = 'D')].apply(lambda x : list(x).count(0),axis=1)

        df_test['kurtosis_for_-{}'.format(i)] = dataframe[pd.date_range(start =(test_date-timedelta(days=i)), periods=i, freq = 'D')].apply(kurtosis, axis=1)

        df_test['hmean_for_-{}'.format(i)] = (dataframe[pd.date_range(start =(test_date-timedelta(days=i)), periods=i, freq = 'D')]+0.0001).apply(hmean, axis=1)

        df_test['skew_for_-{}'.format(i)] = dataframe[pd.date_range(start =(test_date-timedelta(days=i)), periods=i, freq = 'D')].apply(skew, axis=1)

        df_test['mean_for_class_-{}'.format(i)] = train_class_sales_2017[pd.date_range(start =(test_date-timedelta(days=i)), periods=i, freq = 'D')].mean(axis=1).values

        df_test['zero_sales_for_class_-{}'.format(i)] = train_class_sales_2017[pd.date_range(start =(test_date-timedelta(days=i)), periods=i, freq = 'D')].apply(lambda x : list(x).count(0),axis=1).values



    df_test['promosum{}'.format(16)] = test_train_promo_2017[pd.date_range(start =(test_date-timedelta(days=16)), periods=16, freq = 'D')].sum(axis=1)

    df_test['promosum{}'.format(30)] = test_train_promo_2017[pd.date_range(start =(test_date-timedelta(days=30)), periods=30, freq = 'D')].sum(axis=1)

    df_test['promosum{}'.format(60)] = test_train_promo_2017[pd.date_range(start =(test_date-timedelta(days=60)), periods=60, freq = 'D')].sum(axis=1)

    df_test['promosum_class{}'.format(16)] = test_train_class_promo_2017[pd.date_range(start =(test_date-timedelta(days=16)), periods=16, freq = 'D')].sum(axis=1).values

    df_test['promosum_class{}'.format(30)] = test_train_class_promo_2017[pd.date_range(start =(test_date-timedelta(days=30)), periods=30, freq = 'D')].sum(axis=1).values

    df_test['promosum_class{}'.format(60)] = test_train_class_promo_2017[pd.date_range(start =(test_date-timedelta(days=60)), periods=60, freq = 'D')].sum(axis=1).values

  

    

    for i in range(7): #one month worth of day of week values

        df_test['mean_for_dow4_{}'.format(i)] = dataframe[pd.date_range(start =(test_date - timedelta(days=28-i)),periods=4,freq='7D')].mean(axis=1)

        df_test['mean_for_dow8_{}'.format(i)] = dataframe[pd.date_range(start =(test_date - timedelta(days=56-i)),periods=8,freq='7D')].mean(axis=1)

        df_test['mean_for_dow16_{}'.format(i)] = dataframe[pd.date_range(start =(test_date - timedelta(days=112-i)),periods=16,freq='7D')].mean(axis=1)

        df_test['mean_for_dow20_{}'.format(i)] = dataframe[pd.date_range(start =(test_date - timedelta(days=140-i)),periods=20,freq='7D')].mean(axis=1)

        df_test['zero_sales_for_dow_4_{}'.format(i)] = dataframe[pd.date_range(start =(test_date - timedelta(days=28-i)),periods=4,freq='7D')].apply(lambda x : list(x).count(0),axis=1)

        df_test['zero_sales_for_dow_8_{}'.format(i)] = dataframe[pd.date_range(start =(test_date - timedelta(days=56-i)),periods=8,freq='7D')].apply(lambda x : list(x).count(0),axis=1)

        df_test['zero_sales_for_dow_16_{}'.format(i)] = dataframe[pd.date_range(start =(test_date - timedelta(days=112-i)),periods=16,freq='7D')].apply(lambda x : list(x).count(0),axis=1)

        df_test['zero_sales_for_dow_20_{}'.format(i)] = dataframe[pd.date_range(start =(test_date - timedelta(days=140-i)),periods=20,freq='7D')].apply(lambda x : list(x).count(0),axis=1)



    for i in range(16):

        df_test['promo{}'.format(i)] = test_train_promo_2017[test_date+timedelta(days=i)].values.astype(np.uint8)

    #df_test = pd.merge(test.loc[test['date']==test_date],df_test,on=(['item_nbr','store_nbr']),how = 'left').fillna(0)

    return df_test
# use mean,std,var of 30/10 days before to predict the unit_sales

def features_dataset_train(dataframe,train_date):

    

    df_train = []

    y= []

    y = pd.DataFrame(y)

    df_train = pd.DataFrame(df_train)

    month = int(train_date.strftime('%m'))

    week = train_date.isocalendar()[1]

    for i in [10, 20, 30,40,50,60,120]:

        df_train['mean_for_-{}'.format(i)] = dataframe[pd.date_range(start =(train_date-timedelta(days=i)), periods=i,freq = 'D')].mean(axis=1)

        df_train['std_for_-{}'.format(i)] = dataframe[pd.date_range(start =(train_date-timedelta(days=i)), periods=i, freq = 'D')].std(axis=1)

        df_train['var_for_-{}'.format(i)] = dataframe[pd.date_range(start =(train_date-timedelta(days=i)), periods=i, freq = 'D')].var(axis=1)

        df_train['median_for_-{}'.format(i)] = dataframe[pd.date_range(start =(train_date-timedelta(days=i)), periods=i, freq = 'D')].median(axis=1)           

        df_train['zero_sales_for_-{}'.format(i)] = dataframe[pd.date_range(start =(train_date-timedelta(days=i)), periods=i, freq = 'D')].apply(lambda x : list(x).count(0),axis=1)

        df_train['kurtosis_for_-{}'.format(i)] = dataframe[pd.date_range(start =(train_date-timedelta(days=i)), periods=i, freq = 'D')].apply(kurtosis, axis=1)

        df_train['hmean_for_-{}'.format(i)] = (dataframe[pd.date_range(start =(train_date-timedelta(days=i)), periods=i, freq = 'D')]+0.0001).apply(hmean, axis=1)

        df_train['skew_for_-{}'.format(i)] = dataframe[pd.date_range(start =(train_date-timedelta(days=i)), periods=i, freq = 'D')].apply(skew, axis=1)

        df_train['mean_for_class_-{}'.format(i)] = train_class_sales_2017[pd.date_range(start =(train_date-timedelta(days=i)), periods=i, freq = 'D')].mean(axis=1).values

        df_train['zero_sales_for_class_-{}'.format(i)] = train_class_sales_2017[pd.date_range(start =(train_date-timedelta(days=i)), periods=i, freq = 'D')].apply(lambda x : list(x).count(0),axis=1).values

    

    

    df_train['promosum{}'.format(16)] = test_train_promo_2017[pd.date_range(start =(train_date-timedelta(days=16)), periods=16, freq = 'D')].sum(axis=1)

    df_train['promosum{}'.format(30)] = test_train_promo_2017[pd.date_range(start =(train_date-timedelta(days=30)), periods=30, freq = 'D')].sum(axis=1)

    df_train['promosum{}'.format(60)] = test_train_promo_2017[pd.date_range(start =(train_date-timedelta(days=60)), periods=60, freq = 'D')].sum(axis=1)

    df_train['promosum_class{}'.format(16)] = test_train_class_promo_2017[pd.date_range(start =(train_date-timedelta(days=16)), periods=16, freq = 'D')].sum(axis=1).values

    df_train['promosum_class{}'.format(30)] = test_train_class_promo_2017[pd.date_range(start =(train_date-timedelta(days=30)), periods=30, freq = 'D')].sum(axis=1).values

    df_train['promosum_class{}'.format(60)] = test_train_class_promo_2017[pd.date_range(start =(train_date-timedelta(days=60)), periods=60, freq = 'D')].sum(axis=1).values

  

    

    for i in range(7): #one month worth of day of week values

        df_train['mean_for_dow4_{}'.format(i)] = dataframe[pd.date_range(start =(train_date - timedelta(days=28-i)),periods=4,freq='7D')].mean(axis=1)

        df_train['mean_for_dow8_{}'.format(i)] = dataframe[pd.date_range(start =(train_date - timedelta(days=56-i)),periods=8,freq='7D')].mean(axis=1)

        df_train['mean_for_dow16_{}'.format(i)] = dataframe[pd.date_range(start =(train_date - timedelta(days=112-i)),periods=16,freq='7D')].mean(axis=1)

        df_train['mean_for_dow20_{}'.format(i)] = dataframe[pd.date_range(start =(train_date - timedelta(days=140-i)),periods=20,freq='7D')].mean(axis=1)

        df_train['zero_sales_for_dow_4_{}'.format(i)] = dataframe[pd.date_range(start =(train_date - timedelta(days=28-i)),periods=4,freq='7D')].apply(lambda x : list(x).count(0),axis=1)

        df_train['zero_sales_for_dow_8_{}'.format(i)] = dataframe[pd.date_range(start =(train_date - timedelta(days=56-i)),periods=8,freq='7D')].apply(lambda x : list(x).count(0),axis=1)

        df_train['zero_sales_for_dow_16_{}'.format(i)] = dataframe[pd.date_range(start =(train_date - timedelta(days=112-i)),periods=16,freq='7D')].apply(lambda x : list(x).count(0),axis=1)

        df_train['zero_sales_for_dow_20_{}'.format(i)] = dataframe[pd.date_range(start =(train_date - timedelta(days=140-i)),periods=20,freq='7D')].apply(lambda x : list(x).count(0),axis=1)



    for i in range(16):

        y['day{}'.format(i)] = dataframe[(train_date+timedelta(days=i))]

        df_train['promo_{}'.format(i)] = test_train_promo_2017[train_date+timedelta(days=i)].values

    return df_train,y
def find_weights(date):

    weights_tmp= []

    weights_tmp = pd.DataFrame(weights_tmp)

    weights_tmp = pd.merge(pd.DataFrame(train_sales_2017[date]).reset_index(['store_nbr','item_nbr']),items)['perishable'].fillna(0)

    return weights_tmp
import datetime

from datetime import timedelta, date

test_date = date(2017,8,16)

test = pd.read_csv("../input/test.csv")

test['date'] = pd.to_datetime(test['date'])

X_test = []

X_test = features_dataset_test(train_sales_2017,test_date)
import datetime

from datetime import timedelta, date

train_date_gbm = date(2017,7,26)

X_tmp = []

X_train_gbm = []

y_tmp = []

y_train_gbm = []

for i in range(8): #number represents the number of weeks. not sure whats the right number. when i increased from 4 to 6(more data) the score improved. 

    X_tmp,y_tmp = features_dataset_train(train_sales_2017,train_date_gbm) 

    X_train_gbm.append(X_tmp)

    y_train_gbm.append(y_tmp)

    train_date_gbm= train_date_gbm - timedelta(days=7)

    #print(X_train_gbm)

X_train_gbm = pd.concat(X_train_gbm)

y_train_gbm = pd.concat(y_train_gbm)

train_date_gbm = date(2017,7,26)

weight_tmp = []

weight_gbm = []

for i in range(8): #just try with 10 first then increase/decrease accordingly up to u

    weight_tmp = find_weights(train_date_gbm)

    weight_gbm.append(weight_tmp)

    train_date_gbm= train_date_gbm - timedelta(days=7)

weight_gbm = pd.concat(weight_gbm)

val_date_gbm = date(2017,7,26)

weight_val_gbm = []

weight_tmp = find_weights(val_date_gbm)

weight_val_gbm.append(weight_tmp)

weight_val_gbm = pd.concat(weight_val_gbm)

X_val_gbm = []

y_val_gbm = []

val_date_gbm = date(2017,7,26)

X_val_gbm,y_val_gbm = features_dataset_val(train_sales_2017,val_date_gbm)



import lightgbm as lgb

y_test_final = []

params={}

params['max_depth'] = 6

params['learning_rate'] = 0.01

params['boosting_type'] = 'gbdt'

params['objective'] = 'regression'

params['metric'] = 'mse'          

params['bagging_fraction'] = 0.75 

params['feature_fraction'] = 0.75

params['num_leaves'] = 2**6-1        

params['min_data'] = 37          

params['verbose'] = 2

params['num_threads'] = 4

for i in range(16):

    print("Step %d" % (i+1))

    lgb_train = lgb.Dataset(X_train_gbm, label=y_train_gbm.iloc[:,i],weight =np.array(weight_gbm*0.25+1))

    lgb_val = lgb.Dataset(X_val_gbm, label = y_val_gbm.iloc[:,i],reference = lgb_train,weight = np.array(weight_val_gbm*0.25+1))

    lgb_model = lgb.train(params,lgb_train,num_boost_round=2000,valid_sets=[lgb_train,lgb_val],early_stopping_rounds = 50,verbose_eval = 50)

    y_test = lgb_model.predict(X_test,num_iteration = lgb_model.best_iteration)

    y_test_final.append(y_test)

    print(lgb_model.feature_importance('gain'))

    print(y_test)
submission_sales = pd.DataFrame(

np.array(y_test_final).transpose(), index=train_sales_2017.index,

    columns=pd.date_range("2017-08-16", periods=16)).stack().reset_index()

submission_sales = submission_sales.rename(columns = {'level_2' : 'date',0:'unit_sales'})

submission = pd.merge(test,pd.DataFrame(submission_sales),on =['store_nbr','item_nbr','date'],how = 'left').fillna(0)

submission = submission.drop(columns = {'date','store_nbr','item_nbr','onpromotion'})

submission["unit_sales"] = np.expm1(submission["unit_sales"])

submission.loc[submission['unit_sales']<0,['unit_sales']] = 0

submission.to_csv("submission.csv",index=None) #put own path