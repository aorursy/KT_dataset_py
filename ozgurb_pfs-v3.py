# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.preprocessing import LabelEncoder

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

from itertools import product

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.

pd.options.display.float_format = '{:.2f}'.format
items = pd.read_csv('../input/competitive-data-science-predict-future-sales/items.csv')

shops = pd.read_csv('../input/competitive-data-science-predict-future-sales/shops.csv')

cats = pd.read_csv('../input/competitive-data-science-predict-future-sales/item_categories.csv')



train = pd.read_csv('../input/competitive-data-science-predict-future-sales/sales_train.csv')

test  = pd.read_csv('../input/competitive-data-science-predict-future-sales/test.csv')





train = train[train.item_price<100000]

train = train[train.item_cnt_day<1001]

train["item_cnt_day"]=train["item_cnt_day"].fillna(0)



train = train[train.item_cnt_day>0]

median = train[(train.shop_id==32)&(train.item_id==2973)&(train.date_block_num==4)&(train.item_price>0)].item_price.median()

train.loc[train.item_price<0, 'item_price'] = median
def check_nulls(df_train):

    # check distinct values for the columns that contains null value

    # remove nan values that is total count less than 5 percent

    for d in df_train.columns:

        if  df_train[d].isnull().values.any():

            print("column "+d)

            if df_train[d].dtype.kind in 'bifc':

                df_train[d].fillna(0,inplace = True)

            else:

                print("column "+d)

                df_train[d].fillna("NULL_VALUE", inplace = True)
shops['city'] = shops['shop_name'].str.split(' ').map(lambda x: x[0])

shops['city_code'] = LabelEncoder().fit_transform(shops['city'])

cats['split'] = cats['item_category_name'].str.split('-')

cats['type'] = cats['split'].map(lambda x: x[0].strip())

cats['type_code'] = LabelEncoder().fit_transform(cats['type'])

# if subtype is nan then type

cats['subtype'] = cats['split'].map(lambda x: x[1].strip() if len(x) > 1 else x[0].strip())

cats['subtype_code'] = LabelEncoder().fit_transform(cats['subtype'])

cats = cats[['item_category_id','type_code', 'subtype_code']]




# check shop_id pk or not for shops

check_count = shops.groupby(["shop_id"])['shop_id'].size()

print(check_count[check_count > 1])

# check item_category_id pk or not  for cats

check_count = cats.groupby(["item_category_id"])['item_category_id'].size()

print(check_count[check_count > 1])

train.columns
check_count = train.groupby(["date","date_block_num","shop_id","item_id"])['shop_id'].size()

print(check_count[check_count > 1])

train.query("date_block_num==16 and shop_id==50 and item_id==3423")
test["date_block_num"]=34

test = pd.merge(test, shops, on=['shop_id'], how='left',suffixes=('', '_x'))

test = pd.merge(test, items, on=['item_id'], how='left',suffixes=('', '_y'))

test = pd.merge(test, cats, on=['item_category_id'], how='left')



test.columns
check_count = test.groupby(["shop_id","item_id"])['shop_id'].size()

print(check_count[check_count > 1])



# test df PK: shop_id,item_id
# creating a dataframe for per month  cartesian of shop_id*item_id

import itertools

train_2 = []

for i in range(34):

    train_list_1 = train[train.date_block_num==i].date_block_num.unique().tolist()

    train_list_2 = train[train.date_block_num==i].shop_id.unique().tolist()

    train_list_3 = train[train.date_block_num==i].item_id.unique().tolist()



    train_list_f1=list(itertools.product(train_list_1,train_list_2,train_list_3))

    train_2.append(train_list_f1)

# row to columns

train_2 = pd.DataFrame(np.vstack(train_2) )

train_2.rename(columns={0: "date_block_num", 1: "shop_id", 2: "item_id"} , inplace = True)

train_2
cols=['date_block_num','shop_id','item_id']



group = train.groupby(['date_block_num','shop_id','item_id']).agg({'item_cnt_day': ['sum']})

group.columns = ['item_cnt_month']

group.reset_index(inplace=True)



train_2 = pd.merge(train_2, group, on=cols, how='left')

train_2['item_cnt_month'] = (train_2['item_cnt_month']

                                .fillna(0)

                                .clip(0,20) # NB clip target here

                                .astype(np.float32))
train_2 = pd.merge(train_2, shops, on=['shop_id'], how='left',suffixes=('', '_x'))

train_2 = pd.merge(train_2, items, on=['item_id'], how='left',suffixes=('', '_y'))

train_2 = pd.merge(train_2, cats, on=['item_category_id'], how='left')



#train_2["revenue_d"]=train_2["item_price"]*train_2["item_cnt_day"]
check_count = train_2.groupby(["date_block_num","shop_id","item_id"])['shop_id'].size()

print(check_count[check_count > 1])

train_2.describe()
cols = ['shop_id','item_id']

train_2 = pd.concat([train_2, test], ignore_index=True, sort=False, keys=cols)
train_2.describe()
def sum_df(df,columns,target):

    x= [] 

    name=target+"_sum"

    data = columns.split(",")

    for col in data:

        x.append(col)

    print(x)

    return df.groupby(x) [target].sum().to_frame(name).reset_index()



def mean_df(df,columns,target):

    x= [] 

    name=target+"_mean"

    data = columns.split(",")

    for col in data:

        x.append(col)

    print(x)

    return df.groupby(x) [target].mean().to_frame(name).reset_index()



#train_2.describe()

#train_2.rename(columns = {'date_block_num_x':'date_block_num'}, inplace = True) 

def lag_feature(df, lags, col):

    tmp = df[['date_block_num','shop_id','item_id',col]]

    for i in lags:

        shifted = tmp.copy()

        shifted.columns = ['date_block_num','shop_id','item_id', col+'_lag_'+str(i)]

        shifted['date_block_num'] += i

        df = pd.merge(df, shifted, on=['date_block_num','shop_id','item_id'], how='left')

    return df
train_2 = lag_feature(train_2, [1,2,3,6,12], 'item_cnt_month')

train_2.describe()
group = train_2.groupby(['date_block_num']).agg({'item_cnt_month': ['mean']})

group.columns = [ 'date_avg_item_cnt' ]

group.reset_index(inplace=True)



train_2 = pd.merge(train_2, group, on=['date_block_num'], how='left')

train_2['date_avg_item_cnt'] = train_2['date_avg_item_cnt'].astype(np.float16)

train_2 = lag_feature(train_2, [1], 'date_avg_item_cnt')

train_2.drop(['date_avg_item_cnt'], axis=1, inplace=True)





group = train_2.groupby(['date_block_num', 'item_id']).agg({'item_cnt_month': ['mean']})

group.columns = [ 'date_item_avg_item_cnt' ]

group.reset_index(inplace=True)



train_2 = pd.merge(train_2, group, on=['date_block_num','item_id'], how='left')

train_2['date_item_avg_item_cnt'] = train_2['date_item_avg_item_cnt'].astype(np.float16)

train_2 = lag_feature(train_2, [1,2,3,6,12], 'date_item_avg_item_cnt')

train_2.drop(['date_item_avg_item_cnt'], axis=1, inplace=True)





group = train_2.groupby(['date_block_num', 'shop_id']).agg({'item_cnt_month': ['mean']})

group.columns = [ 'date_shop_avg_item_cnt' ]

group.reset_index(inplace=True)



train_2 = pd.merge(train_2, group, on=['date_block_num','shop_id'], how='left')

train_2['date_shop_avg_item_cnt'] = train_2['date_shop_avg_item_cnt'].astype(np.float16)

train_2 = lag_feature(train_2, [1,2,3,6,12], 'date_shop_avg_item_cnt')

train_2.drop(['date_shop_avg_item_cnt'], axis=1, inplace=True)





group = train_2.groupby(['date_block_num', 'item_category_id']).agg({'item_cnt_month': ['mean']})

group.columns = [ 'date_cat_avg_item_cnt' ]

group.reset_index(inplace=True)



train_2 = pd.merge(train_2, group, on=['date_block_num','item_category_id'], how='left')

train_2['date_cat_avg_item_cnt'] = train_2['date_cat_avg_item_cnt'].astype(np.float16)

train_2 = lag_feature(train_2, [1], 'date_cat_avg_item_cnt')

train_2.drop(['date_cat_avg_item_cnt'], axis=1, inplace=True)









group = train_2.groupby(['date_block_num', 'city_code']).agg({'item_cnt_month': ['mean']})

group.columns = [ 'date_city_avg_item_cnt' ]

group.reset_index(inplace=True)



train_2 = pd.merge(train_2, group, on=['date_block_num', 'city_code'], how='left')

train_2['date_city_avg_item_cnt'] = train_2['date_city_avg_item_cnt'].astype(np.float16)

train_2 = lag_feature(train_2, [1], 'date_city_avg_item_cnt')

train_2.drop(['date_city_avg_item_cnt'], axis=1, inplace=True)







group = train_2.groupby(['date_block_num', 'item_id', 'city_code']).agg({'item_cnt_month': ['mean']})

group.columns = [ 'date_item_city_avg_item_cnt' ]

group.reset_index(inplace=True)



train_2 = pd.merge(train_2, group, on=['date_block_num', 'item_id', 'city_code'], how='left')

train_2['date_item_city_avg_item_cnt'] = train_2['date_item_city_avg_item_cnt'].astype(np.float16)

train_2 = lag_feature(train_2, [1], 'date_item_city_avg_item_cnt')

train_2.drop(['date_item_city_avg_item_cnt'], axis=1, inplace=True)







group = train_2.groupby(['date_block_num', 'type_code']).agg({'item_cnt_month': ['mean']})

group.columns = [ 'date_type_avg_item_cnt' ]

group.reset_index(inplace=True)



train_2 = pd.merge(train_2, group, on=['date_block_num', 'type_code'], how='left')

train_2['date_type_avg_item_cnt'] = train_2['date_type_avg_item_cnt'].astype(np.float16)

train_2 = lag_feature(train_2, [1], 'date_type_avg_item_cnt')

train_2.drop(['date_type_avg_item_cnt'], axis=1, inplace=True)









group = train_2.groupby(['date_block_num', 'subtype_code']).agg({'item_cnt_month': ['mean']})

group.columns = [ 'date_subtype_avg_item_cnt' ]

group.reset_index(inplace=True)



train_2 = pd.merge(train_2, group, on=['date_block_num', 'subtype_code'], how='left')

train_2['date_subtype_avg_item_cnt'] = train_2['date_subtype_avg_item_cnt'].astype(np.float16)

train_2 = lag_feature(train_2, [1], 'date_subtype_avg_item_cnt')

train_2.drop(['date_subtype_avg_item_cnt'], axis=1, inplace=True)
group = train_2.groupby(['date_block_num', 'shop_id', 'item_category_id']).agg({'item_cnt_month': ['mean']})

group.columns = ['date_shop_cat_avg_item_cnt']

group.reset_index(inplace=True)



train_2 = pd.merge(train_2, group, on=['date_block_num', 'shop_id', 'item_category_id'], how='left')

train_2['date_shop_cat_avg_item_cnt'] = train_2['date_shop_cat_avg_item_cnt'].astype(np.float16)

train_2 = lag_feature(train_2, [1], 'date_shop_cat_avg_item_cnt')

train_2.drop(['date_shop_cat_avg_item_cnt'], axis=1, inplace=True)
group = train_2.groupby(['date_block_num', 'shop_id', 'item_id']).agg({'item_cnt_month': ['median']})

group.columns = ['date_shop_item_med_item_cnt']

group.reset_index(inplace=True)



train_2 = pd.merge(train_2, group, on=['date_block_num', 'shop_id', 'item_id'], how='left')

train_2['date_shop_item_med_item_cnt'] = train_2['date_shop_item_med_item_cnt'].astype(np.float16)

train_2 = lag_feature(train_2, [1,2,3,6,12], 'date_shop_item_med_item_cnt')

train_2.drop(['date_shop_item_med_item_cnt'], axis=1, inplace=True)





group = train_2.groupby(['date_block_num', 'shop_id']).agg({'item_cnt_month': ['median']})

group.columns = ['date_shop_med_item_cnt']

group.reset_index(inplace=True)



train_2 = pd.merge(train_2, group, on=['date_block_num', 'shop_id'], how='left')

train_2['date_shop_med_item_cnt'] = train_2['date_shop_med_item_cnt'].astype(np.float16)

train_2 = lag_feature(train_2, [1], 'date_shop_med_item_cnt')

train_2.drop(['date_shop_med_item_cnt'], axis=1, inplace=True)
group = train_2.groupby(['date_block_num', 'shop_id','item_category_id']).agg({'item_cnt_month': ['median']})

group.columns = ['date_shop_cat_med_item_cnt']

group.reset_index(inplace=True)



train_2 = pd.merge(train_2, group, on=['date_block_num', 'shop_id','item_category_id'], how='left')

train_2['date_shop_cat_med_item_cnt'] = train_2['date_shop_cat_med_item_cnt'].astype(np.float16)

train_2 = lag_feature(train_2, [1], 'date_shop_cat_med_item_cnt')

train_2.drop(['date_shop_cat_med_item_cnt'], axis=1, inplace=True)
train_2.columns

#train_2.drop(['item_cnt_month_y'], axis=1, inplace=True)
group = train_2.groupby(['date_block_num', 'shop_id','item_id']) ['item_cnt_month'].quantile(.25)

group.columns = ['date_shop_item_q25_item_cnt']

dfx=pd.DataFrame()

dfx["date_shop_item_q25_item_cnt"]=group



print(train_2.columns)

train_2 = pd.merge(train_2, dfx, on=['date_block_num', 'shop_id','item_id'], how='left')

train_2['date_shop_item_q25_item_cnt'] = train_2['date_shop_item_q25_item_cnt'].astype(np.float16)

train_2 = lag_feature(train_2, [1], 'date_shop_item_q25_item_cnt')

train_2.drop(['date_shop_item_q25_item_cnt'], axis=1, inplace=True)
group = train_2.groupby(['date_block_num', 'shop_id','item_id']) ['item_cnt_month'].quantile(.75)

group.columns = ['date_shop_item_q75_item_cnt']

dfx=pd.DataFrame()

dfx["date_shop_item_q75_item_cnt"]=group



print(train_2.columns)

train_2 = pd.merge(train_2, dfx, on=['date_block_num', 'shop_id','item_id'], how='left')

train_2['date_shop_item_q75_item_cnt'] = train_2['date_shop_item_q75_item_cnt'].astype(np.float16)

train_2 = lag_feature(train_2, [1], 'date_shop_item_q75_item_cnt')

train_2.drop(['date_shop_item_q75_item_cnt'], axis=1, inplace=True)
train_2['month'] = train_2['date_block_num'] % 12
def fill_na(df):

    for col in df.columns:

        if ('_lag_' in col) & (df[col].isnull().any()):

            if ('item_cnt' in col):

                df[col].fillna(0, inplace=True)         

    return df



train_2 = fill_na(train_2)
train_2.columns
final=train_2[[

    'date_block_num',

    'shop_id',

    'item_id',

    'item_cnt_month',

    'city_code',

    'item_category_id',

    'type_code',

    'subtype_code',

    

     'item_cnt_month_lag_1', 'item_cnt_month_lag_2', 'item_cnt_month_lag_3', 'item_cnt_month_lag_6', 'date_item_avg_item_cnt_lag_1', 

    'date_item_avg_item_cnt_lag_2', 'date_item_avg_item_cnt_lag_3', 'date_item_avg_item_cnt_lag_6', 'date_cat_avg_item_cnt_lag_1', 

    'date_item_city_avg_item_cnt_lag_1', 'date_subtype_avg_item_cnt_lag_1', 

    'date_shop_cat_avg_item_cnt_lag_1', 'date_shop_item_med_item_cnt_lag_1', 

    'date_shop_item_med_item_cnt_lag_2', 'date_shop_item_med_item_cnt_lag_3', 

    'date_shop_item_med_item_cnt_lag_6', 'date_shop_cat_med_item_cnt_lag_1',

    'date_shop_item_q25_item_cnt_lag_1', 'date_shop_item_q75_item_cnt_lag_1'

            ]]



def numeric_corr(df_train):

    

    # get the list of numeric columns  that has corr more than 0.3 or less than -0.3

    f= df_train.corrwith(df_train.item_cnt_month, axis = 0)

    lister12 = []

    for i,r in f.items():

        if r>0.2 or r<-0.2:

            lister12.append(i)

    return lister12
print( numeric_corr(train_2) )
# check relation between predicted value with all categorical variables

def One_way_ANOVA(df):

    lister12 = []

    for col in df.columns:

        if df[col].dtype.kind not in 'bifc':

            import statsmodels.api as sm

            from statsmodels.formula.api import ols

            model = ols('item_cnt_month ~'+col,data=df).fit()

            table = sm.stats.anova_lm(model, typ=2)

            #print(col)

            #print(table["PR(>F)"][0])

            if table["PR(>F)"][0] < 0.05:

                lister12.append(col)

    return lister12
train_check=train_2[(train_2.date_block_num<=5)]

train_check['item_category_id_char']=train_check['item_category_id'].apply(str)

train_check['type_code_char']=train_check['type_code'].apply(str)

train_check['subtype_code_char']=train_check['subtype_code'].apply(str)
train_check['subtype_code_char']

#train_check["item_category_id"]
One_way_ANOVA(train_check[["city","shop_name","item_category_id_char","subtype_code_char","type_code_char","item_cnt_month"]])
final.rename(columns = {'item_cnt_month':'item_monthly_sum_x'}, inplace = True) 



final["item_monthly_sum_x"] = final["item_monthly_sum_x"]





X_train = final[final.date_block_num < 33].drop(['item_monthly_sum_x'], axis=1)

Y_train = final[final.date_block_num < 33]['item_monthly_sum_x']

X_valid = final[final.date_block_num == 33].drop(['item_monthly_sum_x'], axis=1)

Y_valid = final[final.date_block_num == 33]['item_monthly_sum_x']







X_test = final[final.date_block_num == 34].drop(['item_monthly_sum_x'], axis=1)
final.query('date_block_num == 34')
import xgboost as xgb

import lightgbm as lgb

from sklearn.metrics import mean_squared_error

from math import sqrt

from sklearn.metrics import r2_score







'''

model = xgb.XGBRegressor(    max_depth=8,

    n_estimators=1000,

    min_child_weight=300, 

    colsample_bytree=0.8, 

    subsample=0.8, 

    eta=0.3,    

    seed=42)

'''



model = xgb.XGBRegressor()



model.fit(

    X_train, 

    Y_train, 

    eval_metric="rmse", 

    eval_set=[(X_train, Y_train), (X_valid, Y_valid)], 

    verbose=True, 

    early_stopping_rounds = 10)





""" 

model.fit(X_train, Y_train)

xgb_pred = (model.predict(X_valid).clip(0, 20))



rmse = sqrt(mean_squared_error(Y_valid.clip(0, 20),  xgb_pred) )

print("RMSE RESULT ",rmse)

print("*******")

"""

#1.00553
Y_test = model.predict(X_test)

#print(X_test)





submission = pd.DataFrame({

    "ID": test.ID, 

    "item_cnt_month": Y_test.clip(0, 20)

})

submission.to_csv('xgb_submission.csv', index=False)
