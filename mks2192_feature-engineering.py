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
def featurize_train_test(df):

    # to datetime

    df['first_active_month'] = pd.to_datetime(df['first_active_month'])



    # datetime features

    df['quarter'] = df['first_active_month'].dt.quarter

    df['elapsed_time'] = (datetime.datetime.today() - df['first_active_month']).dt.days



    df['days_feature1'] = df['elapsed_time'] * df['feature_1']

    df['days_feature2'] = df['elapsed_time'] * df['feature_2']

    df['days_feature3'] = df['elapsed_time'] * df['feature_3']



    df['days_feature1_ratio'] = df['feature_1'] / df['elapsed_time']

    df['days_feature2_ratio'] = df['feature_2'] / df['elapsed_time']

    df['days_feature3_ratio'] = df['feature_3'] / df['elapsed_time']





    df['feature_sum'] = df['feature_1'] + df['feature_2'] + df['feature_3']

    df['feature_mean'] = df['feature_sum']/3

    df['feature_max'] = df[['feature_1', 'feature_2', 'feature_3']].max(axis=1)

    df['feature_min'] = df[['feature_1', 'feature_2', 'feature_3']].min(axis=1)

    df['feature_var'] = df[['feature_1', 'feature_2', 'feature_3']].std(axis=1)



    t1 = pd.get_dummies(df_train.feature_1, prefix = 'feature1')

    t2 = pd.get_dummies(df_train.feature_2, prefix = 'feature2')

    t3 = pd.get_dummies(df_train.feature_3, prefix = 'feature3')



    df[t1.columns] = t1

    df[t2.columns] = t2

    df[t3.columns] = t3



    del t1, t2, t3

    gc.collect()



    return df

def Featurized(df1, prefix_string):

    flag = 0

    df1['authorized_flag'] = df1['authorized_flag'].map({'Y':1, "N":0})

    df1['category_1'] = df1['category_1'].map({'Y':1, "N":0})



    df1['category_2'].fillna(1.0,inplace=True)

    df1['category_3'].fillna('A',inplace=True)

    df1['merchant_id'].fillna('M_ID_00a6ca8a8a',inplace=True)

    df1['installments'].replace(-1, np.nan,inplace=True)

    df1['installments'].replace(999, np.nan,inplace=True)



    gb = df1.groupby('card_id')



    df_features = pd.DataFrame()



    df_features['hist_count'] = gb['card_id'].count()



    #authorized_flag

    df_authorized_flag_count = df1.groupby('card_id')['authorized_flag'].value_counts().unstack()

    df_authorized_flag_fraction = np.divide(df_authorized_flag_count, df_authorized_flag_count.sum(axis = 1).values.reshape(-1,1))

    df_authorized_flag_count.columns = df_authorized_flag_count.columns.name +'__' +df_authorized_flag_count.columns.astype('str')+ '_count'

    df_authorized_flag_fraction.columns = df_authorized_flag_fraction.columns.name +'__' +df_authorized_flag_fraction.columns.astype('str')+'_fraction'





    # 'category_1'

    df_category1_count = df1.groupby('card_id')['category_1'].value_counts().unstack()

    df_category1_fraction = np.divide(df_category1_count, df_category1_count.sum(axis = 1).values.reshape(-1,1))

    df_category1_count.columns = df_category1_count.columns.name +'__' +df_category1_count.columns.astype('str')+ '_count'

    df_category1_fraction.columns = df_category1_fraction.columns.name +'__' +df_category1_fraction.columns.astype('str')+'_fraction'





    # 'category_2'

    df_category2_count = df1.groupby('card_id')['category_2'].value_counts().unstack()

    df_category2_fraction = np.divide(df_category2_count, df_category2_count.sum(axis = 1).values.reshape(-1,1))

    df_category2_count.columns = df_category2_count.columns.name +'__' +df_category2_count.columns.astype('str')+ '_count'

    df_category2_fraction.columns = df_category2_fraction.columns.name +'__' +df_category2_fraction.columns.astype('str')+'_fraction'





    # 'category_3'

    df_category3_count = df1.groupby('card_id')['category_3'].value_counts().unstack()

    df_category3_fraction = np.divide(df_category3_count, df_category3_count.sum(axis = 1).values.reshape(-1,1))

    df_category3_count.columns = df_category3_count.columns.name +'__' +df_category3_count.columns.astype('str')+ '_count'

    df_category3_fraction.columns =  df_category3_fraction.columns.name +'__' + df_category3_fraction.columns.astype('str')+'_fraction'





    #'city_id'

        #bin creation

    city_id_count = df1.groupby('city_id')['city_id'].count()

    np.log(city_id_count).hist()

    bins = pd.qcut(np.log(city_id_count), 5, duplicates='drop')

    df1['city_id_bins'] = df1['city_id'].map(bins)



        #column creation (count and fraction)

    df_city_id_count = df1.groupby('card_id')['city_id_bins'].value_counts().unstack()

    df_city_id_fraction = np.divide(df_city_id_count, df_city_id_count.sum(axis = 1).values.reshape(-1,1))

    df_city_id_count.columns = df_city_id_count.columns.name +'__' +df_city_id_count.columns.astype('str')+ '_count'

    df_city_id_fraction.columns = df_city_id_fraction.columns.name +'__' + df_city_id_fraction.columns.astype('str')+'_fraction'





    #'merchant_category_id'

        #bin creation

    merchant_category_id_count = df1.groupby('merchant_category_id')['merchant_category_id'].count()

    np.log(merchant_category_id_count).hist()

    bins = pd.qcut(np.log(merchant_category_id_count), 5,duplicates='drop')

    df1['merchant_category_id_bins'] = df1['merchant_category_id'].map(bins)



        #column creation (count and fraction)

    df_merchant_category_id_count = df1.groupby('card_id')['merchant_category_id_bins'].value_counts().unstack()

    df_merchant_category_id_fraction = np.divide(df_merchant_category_id_count, df_merchant_category_id_count.sum(axis = 1).values.reshape(-1,1))

    df_merchant_category_id_count.columns = df_merchant_category_id_count.columns.name +'__' +df_merchant_category_id_count.columns.astype('str')+ '_count'

    df_merchant_category_id_fraction.columns = df_merchant_category_id_fraction.columns.name +'__' + df_merchant_category_id_fraction.columns.astype('str')+'_fraction'





    #'merchant_id'

        #bin creation

    merchant_id_count = df1.groupby('merchant_id')['merchant_id'].count()

    np.log(merchant_id_count).hist()

    bins = pd.qcut(np.log(merchant_id_count), 5,duplicates='drop')

    df1['merchant_id_bins'] = df1['merchant_id'].map(bins)



        #column creation (count and fraction)

    df_merchant_id_count = df1.groupby('card_id')['merchant_id_bins'].value_counts().unstack()

    df_merchant_id_fraction = np.divide(df_merchant_id_count, df_merchant_id_count.sum(axis = 1).values.reshape(-1,1))

    df_merchant_id_count.columns = df_merchant_id_count.columns.name +'__' +df_merchant_id_count.columns.astype('str')+ '_count'

    df_merchant_id_fraction.columns = df_merchant_id_fraction.columns.name +'__' + df_merchant_id_fraction.columns.astype('str')+'_fraction'





    #'state_id'

        #bin creation

    state_id_count = df1.groupby('state_id')['state_id'].count()

    np.log(state_id_count).hist()

    bins = pd.qcut(np.log(state_id_count), 5,duplicates='drop')

    df1['state_id_bins'] = df1['state_id'].map(bins)



        #column creation (count and fraction)

    df_state_id_count = df1.groupby('card_id')['state_id_bins'].value_counts().unstack()

    df_state_id_fraction = np.divide(df_state_id_count, df_state_id_count.sum(axis = 1).values.reshape(-1,1))

    df_state_id_count.columns = df_state_id_count.columns.name +'__' +df_state_id_count.columns.astype('str')+ '_count'

    df_state_id_fraction.columns = df_state_id_fraction.columns.name +'__' + df_state_id_fraction.columns.astype('str')+'_fraction'





    #'subsector_id'

        #bin creation

    subsector_id_count = df1.groupby('subsector_id')['subsector_id'].count()

    np.log(subsector_id_count).hist()

    bins = pd.qcut(np.log(subsector_id_count), 5, duplicates='drop')

    df1['subsector_id_bins'] = df1['subsector_id'].map(bins)



        #column creation (count and fraction)

    df_subsector_id_count = df1.groupby('card_id')['subsector_id_bins'].value_counts().unstack()

    df_subsector_id_fraction = np.divide(df_subsector_id_count, df_subsector_id_count.sum(axis = 1).values.reshape(-1,1))

    df_subsector_id_count.columns = df_subsector_id_count.columns.name +'__' +df_subsector_id_count.columns.astype('str')+ '_count'

    df_subsector_id_fraction.columns = df_subsector_id_fraction.columns.name +'__' + df_subsector_id_fraction.columns.astype('str')+'_fraction'





    # 'installments', 'month_lag', 'purchase_amount'



    Min = df1.groupby('card_id')[['installments', 'month_lag', 'purchase_amount']].min()

    Min.columns = np.array(Min.columns)+'_mean'

    Max = df1.groupby('card_id')[['installments', 'month_lag', 'purchase_amount']].max()

    Max.columns = np.array(Max.columns)+'_max'

    Median = df1.groupby('card_id')[['installments', 'month_lag', 'purchase_amount']].median()

    Median.columns = np.array(Median.columns)+'_median'

    Std = df1.groupby('card_id')[['installments', 'month_lag', 'purchase_amount']].std()

    Std.columns = np.array(Std.columns)+'_std'

    Skew = df1.groupby('card_id')[['installments', 'month_lag', 'purchase_amount']].skew()

    Skew.columns = np.array(Skew.columns)+'_skew'

    Mad =df1.groupby('card_id')[['installments', 'month_lag', 'purchase_amount']].mad()

    Mad.columns = np.array(Mad.columns)+'_mad'

    Sum =df1.groupby('card_id')[['installments', 'month_lag', 'purchase_amount']].sum()

    Sum.columns = np.array(Sum.columns)+'_sum'





    # 'purchase_date'



    df_features["purchase_date_max"] = df1.groupby('card_id')['purchase_date'].max()

    df_features["purchase_date_min"] = df1.groupby('card_id')['purchase_date'].min()



    #df_features['first_buy'] = (df_features['purchase_date_min'] - df['first_active_month']).dt.days



    df1['today_purchase_date'] =  pd.datetime.today() - pd.to_datetime(df1.purchase_date)

    df1['purchase_date_month_diff'] = df1['today_purchase_date'].dt.total_seconds()/(3600*24*30) - df1.month_lag





    gb = df1.groupby('card_id')['purchase_date_month_diff'].apply(sorted).apply(np.diff)



    try:

        mean = gb.apply(np.mean).rename("purchase_date_month_diff"+'_mean')

        median = gb.apply(np.median).rename("purchase_date_month_diff"+'_median')

        std = gb.apply(np.std).rename("purchase_date_month_diff"+'_std')

        max1 = gb.apply(np.max).rename("purchase_date_month_diff"+'_max')

        min1 = gb.apply(np.min).rename("purchase_date_month_diff"+'_min')

        sum1 = gb.apply(np.sum).rename("purchase_date_month_diff"+'_sum')

    except:

        flag = 1



    #=============================== Appending into One File ================



    #pd.DataFrame(df_features['old_hist_count']).to_csv("appended.csv", index=True)



    List = [df_features['hist_count'], df_authorized_flag_count, df_authorized_flag_fraction, df_category1_count,df_category1_fraction,

               df_category2_count, df_category2_fraction, df_category3_count, df_category3_fraction,

               df_city_id_count, df_city_id_fraction, df_merchant_category_id_count, df_merchant_category_id_fraction,

               df_merchant_id_count, df_merchant_id_fraction, df_state_id_count, df_state_id_fraction,

               df_subsector_id_count, df_subsector_id_fraction, Min, Max, Median, Std, Skew, Mad, Sum]

               

    if(flag !=1):

        List = List+[mean, median, std, max1, min1, sum1]    



    df_concat = pd.concat(List, axis = 1, ignore_index=False)

    df_concat.columns = prefix_string + np.array(df_concat.columns)



    ##df_concat.to_csv("appended.csv")



    for i in range(1):

        # datetime features

        df1['purchase_date'] = pd.to_datetime(df1['purchase_date'])

        df1['month'] = df1['purchase_date'].dt.month

        df1['day'] = df1['purchase_date'].dt.day

        df1['hour'] = df1['purchase_date'].dt.hour

        df1['weekofyear'] = df1['purchase_date'].dt.weekofyear

        df1['weekday'] = df1['purchase_date'].dt.weekday

        df1['weekend'] = (df1['purchase_date'].dt.weekday >=5).astype(int)



        # additional features

        df1['price'] = df1['purchase_amount'] / df1['installments']



        #Christmas : December 25 2017

        df1['Christmas_Day_2017']=(pd.to_datetime('2017-12-25')-df1['purchase_date']).dt.days.apply(lambda x: x if x > 0 and x < 100 else 0)

        #Mothers Day: May 14 2017

        df1['Mothers_Day_2017']=(pd.to_datetime('2017-06-04')-df1['purchase_date']).dt.days.apply(lambda x: x if x > 0 and x < 100 else 0)

        #fathers day: August 13 2017

        df1['fathers_day_2017']=(pd.to_datetime('2017-08-13')-df1['purchase_date']).dt.days.apply(lambda x: x if x > 0 and x < 100 else 0)

        #Childrens day: October 12 2017

        df1['Children_day_2017']=(pd.to_datetime('2017-10-12')-df1['purchase_date']).dt.days.apply(lambda x: x if x > 0 and x < 100 else 0)

        #Valentine's Day : 12th June, 2017

        df1['Valentine_Day_2017']=(pd.to_datetime('2017-06-12')-df1['purchase_date']).dt.days.apply(lambda x: x if x > 0 and x < 100 else 0)

        #Black Friday : 24th November 2017

        df1['Black_Friday_2017']=(pd.to_datetime('2017-11-24') - df1['purchase_date']).dt.days.apply(lambda x: x if x > 0 and x < 100 else 0)



        #2018

        #Mothers Day: May 13 2018

        df1['Mothers_Day_2018']=(pd.to_datetime('2018-05-13')-df1['purchase_date']).dt.days.apply(lambda x: x if x > 0 and x < 100 else 0)



        df1['month_diff'] = ((datetime.datetime.today() - df1['purchase_date']).dt.days)//30

        df1['month_diff'] += df1['month_lag']



        # additional features

        df1['duration'] = df1['purchase_amount']*df1['month_diff']

        df1['amount_month_ratio'] = df1['purchase_amount']/df1['month_diff']



    col_seas = ['month', 'hour', 'weekofyear', 'weekday', 'day']

    aggs = {}

    

    for col in col_seas:

        aggs[col] = ['nunique', 'mean', 'min', 'max']



    for i in range(1):

        aggs['purchase_date'] = ['max','min']

        aggs['weekend'] = ['mean']

        aggs['month'] = ['mean', 'min', 'max']

        aggs['weekday'] = ['mean', 'min', 'max']

        aggs['price'] = ['mean','max','min','var']

        aggs['Christmas_Day_2017'] = ['mean']

        aggs['Children_day_2017'] = ['mean']

        aggs['Black_Friday_2017'] = ['mean']

        aggs['Mothers_Day_2018'] = ['mean']

        aggs['duration']=['mean','min','max','var','skew']

        aggs['amount_month_ratio']=['mean','min','max','var','skew']





    df_temp = df1.groupby('card_id').agg(aggs)



    # change column name

    df_temp.columns = pd.Index([e[0] + "_" + e[1] for e in df_temp.columns.tolist()])

    df_temp.columns = [prefix_string+ c for c in df_temp.columns]



    if(prefix_string =='old_'):

        df_temp['old_purchase_date_diff'] = (df_temp['old_purchase_date_max']-df_temp['old_purchase_date_min']).dt.days

        #df_temp['hist_purchase_date_average'] = df_temp['hist_purchase_date_diff']/df_temp['hist_card_id_size']

        df_temp['old_purchase_date_uptonow'] = (datetime.datetime.today()-df_temp['old_purchase_date_max']).dt.days

        df_temp['old_purchase_date_uptomin'] = (datetime.datetime.today()-df_temp['old_purchase_date_min']).dt.days



    if(prefix_string =='new_'):

        df_temp['new_purchase_date_diff'] = (df_temp['new_purchase_date_max']-df_temp['new_purchase_date_min']).dt.days

        #df_temp['hist_purchase_date_average'] = df_temp['hist_purchase_date_diff']/df_temp['hist_card_id_size']

        df_temp['new_purchase_date_uptonow'] = (datetime.datetime.today()-df_temp['new_purchase_date_max']).dt.days

        df_temp['new_purchase_date_uptomin'] = (datetime.datetime.today()-df_temp['new_purchase_date_min']).dt.days



    if(prefix_string =='old_new_'):

        df_temp['old_new_purchase_date_diff'] = (df_temp['old_new_purchase_date_max']-df_temp['old_new_purchase_date_min']).dt.days

        #df_temp['hist_purchase_date_average'] = df_temp['hist_purchase_date_diff']/df_temp['hist_card_id_size']

        df_temp['old_new_purchase_date_uptonow'] = (datetime.datetime.today()-df_temp['old_new_purchase_date_max']).dt.days

        df_temp['old_new_purchase_date_uptomin'] = (datetime.datetime.today()-df_temp['old_new_purchase_date_min']).dt.days





    df_concat_new = pd.concat([df_concat,df_temp], axis = 1, ignore_index=False)



    return df_concat_new



Path = "../input/"

import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

import datetime

from tqdm import tqdm

import gc



df1 = pd.read_csv(Path+"historical_transactions.csv")

df2 = pd.read_csv(Path+"merchants.csv")

df3 = pd.read_csv(Path+"new_merchant_transactions.csv")

df_train = pd.read_csv(Path+"train.csv")

df_test = pd.read_csv(Path+"test.csv")
#A = B



#--- Feature Engineering ---------------



##authorized_flag                2  (cat)    #done

##card_id                   325540  (cat uid)   #done

##city_id                      308   (cat)   #done

##category_1                     2    (cat)   #done

##installments                  15    (cat/numeric) #done

##category_3                     3    (cat)     #done

##merchant_category_id         327     (cat)    #done

##merchant_id               326311      (cat)   #done

##month_lag                     14      (cat/numeric)  #done

##purchase_amount           215014     (numeric)      #done

##purchase_date           16395300     (cat and numeric)

##category_2                     5     (cat)      #done

##state_id                      25      (cat)       #done

##subsector_id                  41      (cat)       #done

##dtype: int64







##authorized_flag         object

##card_id                 object

##city_id                 int64

##category_1              object

##installments            int64

##category_3              object

##merchant_category_id    int64

##merchant_id             object

##month_lag               int64

##purchase_amount         float64

##purchase_date           object

##category_2              float64

##state_id                int64

##subsector_id            int64



#df3.to_csv("old_and_new.csv", mode='a', columns=False, index=False)



Columns  = ['authorized_flag', 'card_id', 'city_id', 'category_1', 'installments',

       'category_3', 'merchant_category_id', 'merchant_id', 'month_lag',

       'purchase_amount', 'purchase_date', 'category_2', 'state_id',

       'subsector_id']



print("This takes more than the kernal time limit Do run on your PC")

print("Or may be try to optimize the Code")

print("Output files are shared below")

# 

print("https://www.kaggle.com/mks2192/elo-features-data-7-feb")



A = B #Exiting the code



df_old = Featurized(df1, 'old_')

df_new = Featurized(df3, 'new_')



del df1, df3

gc.collect()



# for i in range(1):

#     temp = pd.read_csv("old_and_new.csv")

#     df_appended = Featurized(temp, 'old_new_')



df_old_new = pd.concat([df_old,df_new], axis = 1, ignore_index=False)

df_old_new = df_old_new.reset_index().rename(columns = {'index':'card_id'})



#================ Featurized train and test =================



train = featurize_train_test(df_train)

test = featurize_train_test(df_test)



#================================== train and test ==========================



train_df = pd.merge(df_old_new, train , on ='card_id')

test_df = pd.merge(df_old_new, test , on ='card_id')



#============================================================================

train_df.to_csv("train.csv")

test_df.to_csv("test.csv")
