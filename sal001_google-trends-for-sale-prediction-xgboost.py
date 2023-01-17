!pip install googletrans

!pip install pytrends
#import require modules

import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

import xgboost as xgb

%matplotlib inline



from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import KFold

from sklearn.metrics import mean_squared_error



import gc

import pickle

from sys import getsizeof

from googletrans import Translator

from pytrends.request import TrendReq
#variables setting

pd.set_option('display.max_columns', 100)

pd.set_option('display.max_rows', 100)

ori_data_path = '/kaggle/input/competitive-data-science-predict-future-sales/'

model_path = '/'

processed_data_path = '/'

tables_path = '/'

items_IP_path = '../input/predict-future-sales-ip-groups/items_IPs_v1.csv'

items_IP_trend_path = '../input/predict-future-sales-ip-groups/items_IP_trend.csv'
#import datasets

df = pd.read_csv(ori_data_path+'sales_train.csv')

df_test = pd.read_csv(ori_data_path+'test.csv')

sample_submission = pd.read_csv(ori_data_path+'sample_submission.csv')

item_categories = pd.read_csv(ori_data_path+'item_categories.csv')

items = pd.read_csv(ori_data_path+'items.csv')

shops = pd.read_csv(ori_data_path+'shops.csv')
df[['date_block_num','shop_id']] = df[['date_block_num','shop_id']].astype('int8')

df[['item_cnt_day','item_id']] = df[['item_cnt_day','item_id']].astype('int16')

df['item_price'] = df['item_price'].astype('float32')
shop_test = df_test['shop_id'].nunique()

item_test = df_test['item_id'].nunique()

total_product_test = df_test[['shop_id','item_id']].drop_duplicates()

shop_train = df['shop_id'].nunique()

item_train = df['item_id'].nunique()

total_product_train = df[['shop_id','item_id']].drop_duplicates()



print('We need to predict products from {} shops and {} kinds of products (total shop-item {} pairs)'.format(shop_test,item_test,total_product_test.shape[0]))

print('given that historical daily sale number from {} shops and {} kinds of products (total shop-item {} pairs)'.format(shop_train,item_train,total_product_train.shape[0]))
#100,000 pairs have no sale recored

common_pairs = total_product_test.merge(total_product_train,how='inner',on=['shop_id','item_id']).shape[0]

test_only_pairs = total_product_test.shape[0] - common_pairs

print(f'{common_pairs} pairs of products are shared by historical and to be predicted.')

print(f'{test_only_pairs} pairs are new pairs, so there is no corresponding historical data for prediciton.')
#363 items are new items that never seen before

total_items_test = total_product_test['item_id'].drop_duplicates()

total_items_train = total_product_train['item_id'].drop_duplicates()

total_items_join = pd.merge(total_items_test,total_items_train,how='outer',on=['item_id'],indicator=True)

total_items_join = total_items_join.replace({'both':'hisory and target','left_only':'target only','right_only':'hisory only'})

total_items_join['_merge'].value_counts()
del shop_test

del item_test

del total_product_test

del shop_train

del item_train

del total_product_train

del common_pairs

del test_only_pairs

del total_items_test

del total_items_train

del total_items_join

gc.collect()
shop_item_pair_num_month = df

shop_item_pair_num_month['month'] = shop_item_pair_num_month['date'].apply(lambda d: d.split('.')[2] + '-' + d.split('.')[1])

shop_item_pair_num_month = df.groupby(['month','date_block_num'])['item_cnt_day'].sum()

shop_item_pair_num_month.name = 'monthly_sale_cnt'

shop_item_pair_num_month = shop_item_pair_num_month.reset_index()

ax = shop_item_pair_num_month.plot(x='month',y='monthly_sale_cnt',x_compat=True,figsize=(15,4),marker='o')



plt.xticks(shop_item_pair_num_month.date_block_num, shop_item_pair_num_month.month, rotation='vertical')



plt.xlabel('Year-Month',fontsize=16)

plt.ylabel('Monthly Sale Count (ea)',fontsize=16)

plt.title('Monthly Sale Count Trend',fontsize=20)
all_shops = np.sort(df['shop_id'].unique())

all_items = np.sort(df['item_id'].unique())

all_pairs = pd.DataFrame(index=all_shops,columns=all_items)

all_pairs = all_pairs.fillna(-1).stack()

all_pairs.index.names = ['shop_id','item_id']

all_pairs = all_pairs.reset_index()

all_pairs = all_pairs.drop(0,axis=1)



all_existed_pair_month = df.pivot_table(values='item_cnt_day', columns='date_block_num', index=['shop_id','item_id'],aggfunc='sum')



all_pairs = pd.merge(all_pairs,all_existed_pair_month,on=['shop_id','item_id'],how='left')



plt.figure(figsize=(16,12))



sns.heatmap(all_pairs.groupby('shop_id').sum().drop('item_id',axis=1).replace(0,np.nan),vmax=5000,cmap='plasma',linewidths=1,linecolor='black')

plt.xticks(shop_item_pair_num_month.date_block_num, shop_item_pair_num_month.month, rotation='vertical')



plt.xlabel('Year-Month',fontsize=16)

plt.ylabel('Shops',fontsize=16)

plt.title('Shops Monthly Sale',fontsize=20)
item_trend = df.pivot_table(index='date',columns='item_id',values='item_cnt_day',aggfunc='mean').fillna(0)

#original date format is dd.mm.yyyy

item_trend.index = item_trend.index.map(lambda d:pd.to_datetime(d.split('.')[2]+'.'+d.split('.')[1]+'.'+d.split('.')[0]))



#set multiple plots

fig = plt.figure(figsize=(15,6))

ax1 = fig.add_subplot(311)

ax1.set(ylabel='Sale Count (ea)')

ax1.set_title('best seller shopping bags(20949)')



ax2 = fig.add_subplot(312)

ax2.set(ylabel='Sale Count (ea)')

ax2.set_title('Battery(22088)')



ax3 = fig.add_subplot(313)

ax3.set(ylabel='Sale Count (ea)')

ax3.set_title('PlayStation Network wallet (5823)')

plt.tight_layout()





#best seller shopping bags

item_1 = item_trend[20949].sort_index()

item_1 = pd.DataFrame({'day':item_1,

                         '7days_avg':item_1.rolling(window=7).mean(),

                         '30days_avg':item_1.rolling(window=30).mean()})

#Battery

item_2 = item_trend[22088].sort_index()

item_2 = pd.DataFrame({'day':item_2,

                          '7days_avg':item_2.rolling(window=7).mean(),

                          '30days_avg':item_2.rolling(window=30).mean()})

#PlayStation Network wallet

item_3 = item_trend[5823].sort_index()

item_3 = pd.DataFrame({'day':item_3,

                            '7days_avg':item_3.rolling(window=7).mean(),

                            '30days_avg':item_3.rolling(window=30).mean()})





sns.lineplot(data=item_1, palette='dark', linewidth=3, ax=ax1)

sns.lineplot(data=item_2, palette='dark', linewidth=3, ax=ax2)

sns.lineplot(data=item_3, palette='dark', linewidth=3, ax=ax3)

#set multiple plots

fig = plt.figure(figsize=(15,6))

ax1 = fig.add_subplot(311)

ax1.set(ylabel='Sale Count (ea)')

ax1.set_title('Movie "Kingsman: The Secret Service" DVD (4084)')



ax2 = fig.add_subplot(312)

ax2.set(ylabel='Sale Count (ea)')

ax2.set_title('Grand Theft Auto V PS3 (3732)')



ax3 = fig.add_subplot(313)

ax3.set(ylabel='Sale Count (ea)')

ax3.set_title('annual exhibition Ticket "Igromir 2014"(9241)')



plt.tight_layout()





#Movie 'Kingsman' DVD , release day Jun 2015

item_1 = item_trend[4084].sort_index()

item_1 = pd.DataFrame({'day':item_1,

                         '7days_avg':item_1.rolling(window=7).mean(),

                         '30days_avg':item_1.rolling(window=30).mean()})

#Grand Theft Auto V PS3 (highest volumne at beginning), Sep 2013  Release on PS3

item_2 = item_trend[3732].sort_index()

item_2 = pd.DataFrame({'day':item_2,

                          '7days_avg':item_2.rolling(window=7).mean(),

                          '30days_avg':item_2.rolling(window=30).mean()})

#annual exhibition Ticket "Igromir 2014"  (October 3, 2014), cut off after event started

item_3 = item_trend[9241].sort_index()

item_3 = pd.DataFrame({'day':item_3,

                            '7days_avg':item_3.rolling(window=7).mean(),

                            '30days_avg':item_3.rolling(window=30).mean()})





sns.lineplot(data=item_1, palette='dark', linewidth=3, ax=ax1)

sns.lineplot(data=item_2, palette='dark', linewidth=3, ax=ax2)

sns.lineplot(data=item_3, palette='dark', linewidth=3, ax=ax3)

del shop_item_pair_num_month

del ax

del all_pairs

del all_existed_pair_month

del all_items

del all_shops

del item_trend

del item_1

del item_2

del item_3

del fig

del ax1

del ax2

del ax3

gc.collect()
%%time

for m in df['date_block_num'].unique():

    shops_list_m = np.sort(df[df['date_block_num']==m]['shop_id'].unique())

    items_list_m = np.sort(df[df['date_block_num']==m]['item_id'].unique())

    all_pairs_m = pd.DataFrame(index=shops_list_m,columns=items_list_m).fillna(m).stack()

    all_pairs_m.index.names = ['shop_id','item_id']

    all_pairs_m.name = 'date_block_num'

    

    assert len(shops_list_m) * len(items_list_m) == len(all_pairs_m)

    

    if m == 0:

        dataset = all_pairs_m

    else:

        dataset = pd.concat([dataset,all_pairs_m],axis=0)

        

    del shops_list_m

    del items_list_m

    del all_pairs_m

    gc.collect()

    

dataset = dataset.reset_index()

dataset = dataset.sort_values(by=['shop_id','item_id','date_block_num']).reset_index(drop=True)
cnt_month = df.groupby(['shop_id','item_id','date_block_num']).sum()['item_cnt_day']

cnt_month.name = 'item_cnt_month'



dataset = dataset.merge(cnt_month,on=['shop_id','item_id','date_block_num'],how='left')

dataset['item_cnt_month'] = dataset['item_cnt_month'].fillna(0).clip(0,20)



del cnt_month

gc.collect()
price_pivot_month = df.pivot_table(values='item_price', columns='date_block_num', index=['shop_id','item_id'])



price_pivot_month = price_pivot_month.fillna(method='ffill',axis=1)

price_pivot_month = price_pivot_month.fillna(0)



#transform from wide to long format

price_pivot_month = price_pivot_month.stack()

price_pivot_month.name = 'avg_price'

price_pivot_month = price_pivot_month.reset_index()



#clip price to 0~50000 to prevent outliers

price_pivot_month['avg_price'] = price_pivot_month['avg_price'].clip(0,50000)



#merge back to main table

dataset = dataset.merge(price_pivot_month,on=['shop_id','item_id','date_block_num'],how='left').fillna(0)

dataset['revenue']=dataset['avg_price']*dataset['item_cnt_month']



del price_pivot_month

gc.collect()
df_test['date_block_num'] = 34

dataset = pd.concat([dataset,df_test.drop('ID',axis=1)],axis=0)

dataset = dataset.sort_values(by=['date_block_num','shop_id','item_id']).reset_index(drop=True)



assert dataset.isnull().sum().max() == len(df_test)



dataset = dataset.fillna(0) #only for test data
#compress DataFrame for lower memory usage

float32_cols = ['revenue','avg_price']

int16_cols =['item_id']

int8_cols = ['date_block_num','shop_id','item_cnt_month']



dataset[int8_cols] = dataset[int8_cols].astype('int8')

dataset[int16_cols] = dataset[int16_cols].astype('int16')

dataset[float32_cols] = dataset[float32_cols].astype('float32')
def add_lag_v3(dataset,lookup_set,lag_months,lag_features_list,on_key,fillna_value,dtypes):

    

    """

    Function to create one of multiple lagged features

    

    Parameters: 

    ------------

    dataset : (pd.DataFrame) target dataset 

    

    look_set :(pd.DataFrame)  source of features. Need to be groupped by desired key such as shop_id, item_category_id, or 

        other customized group. create_mean_encoding_table() is created for this. 'None' means using the target dataset. 

    

    lag_months : (list) list of lagged features, ex. [1,2,3] means adding 3 columns with lagged 1,2 and 3

    

    lag_features_list : (list) list of targeted features' name, should be existing in the lookup_set

    

    on_key : (list) list of keys used to merge the dataset and lookup_set

    

    fillna_value : (list) how to fill missing for each of lagged features in the lag_features_list.

    

    dtypes : (list) dtypes of generated lagged featrures. Note 'fillna_value' and 'dtypese' are addtional properties of 

        'lag_features_list', so it is necessery to be in the same order and length. 

    ------------

    """

    if lookup_set is None:

        lookup_set = dataset[on_key + lag_features_list].copy()

    else:    

        dataset = pd.merge(dataset,lookup_set,how='left',on=on_key)

    

    #breakpoint()

    

    

    # create lagged features for each month

    for lag in lag_months:

        dataset_alignment = dataset[on_key + lag_features_list].copy() #small subset for alignment to reduce memory usage

        dataset_lag = lookup_set[on_key + lag_features_list]

        dataset_lag['date_block_num'] = dataset_lag['date_block_num'] + lag

        dataset_alignment = pd.merge(dataset_alignment,dataset_lag,how='left',on=on_key,suffixes=['','_lag_'+str(lag)+'m'])

        dataset_alignment = dataset_alignment.drop(on_key + lag_features_list,axis=1)

        #breakpoint()

        dataset = pd.concat([dataset,dataset_alignment],axis=1,join='outer')

        del dataset_lag

        del dataset_alignment

        gc.collect()

    

    # dtypes setting

    for i,feature in enumerate(lag_features_list):

        new_columns = dataset.columns.str.startswith(feature + '_lag_')

        dataset.loc[:,new_columns] = dataset.loc[:,new_columns].fillna(fillna_value[i]).astype(dtypes[i])

        

    return dataset
def add_first_last_sale(dataset,source_set,keys,prefix):

    """

    Search the month of first sale and last sale, only read historical data, 

    ex. for month 8, only month 0~7 will be used to judge.

    Then, it creates both relative and absolut values. 

    ex. at month 8, the absolute last month is 7, and then the relative will be -1

    

    Parameters: 

    ------------

    dataset : (pd.DataFrame) target dataset 

    

    source_set :(pd.DataFrame) source dataframe to judge first/last

    

    keys : (list) level of group. ex. ['shop_id','item_id'] means judging by shop-item pair 

    

    prefix : (str) prefix of created columns

    ------------

    """    

    for m in range(1,35): #1~34m

    

        #only use historical data to judge (0 to m-1)

        tmp = source_set[source_set['date_block_num']<m].drop_duplicates(keys+['date_block_num']).groupby(keys).agg({'date_block_num':[np.min,np.max]})

        tmp.columns = [prefix + '_first_sold', prefix + '_last_sold']

        tmp = tmp.reset_index()

        tmp['date_block_num'] = m 

        

        if m == 1:  

            first_last_sold = tmp

        else:

            first_last_sold = pd.concat([first_last_sold,tmp],axis=0)

            

        del tmp

        gc.collect()

    #breakpoint()

        

    dataset = dataset.merge(first_last_sold,on=keys+['date_block_num'],how='left')

        

    dataset[prefix + '_first_sold_offset'] = dataset[prefix + '_first_sold'] - dataset['date_block_num']

    dataset[prefix + '_last_sold_offset'] = dataset[prefix + '_last_sold'] - dataset['date_block_num']

    

    columns = [prefix + '_first_sold', prefix + '_last_sold',prefix + '_first_sold_offset', prefix + '_last_sold_offset']

    dataset[columns] = dataset[columns].fillna(0)

    dataset[columns] = dataset[columns].astype('int8')

    

    return dataset
def price_diff(row,columnA,columnB):

        #compare columnA price with columnB ((A-B)/B) in percentage

        if row[columnB] == 0:

            increase_percent = 0

        else:    

            increase_percent = round((row[columnA]-row[columnB])/row[columnB]*100)

        return increase_percent
def create_mean_encoding_table(dataset:pd.DataFrame, encode_item:list, prefix:str):

    #breakpoint()

    encoding_table = dataset[encode_item +['date_block_num','item_cnt_month','revenue','avg_price']] #remove 0 to reduce bias

    encoding_table.loc[:,['item_cnt_month','revenue','avg_price']] = encoding_table.loc[:,['item_cnt_month','revenue','avg_price']].replace(0,np.nan)

    encoding_table = encoding_table.groupby(encode_item + ['date_block_num']).agg({'item_cnt_month':[np.mean],'revenue':[np.mean],'avg_price':[np.mean]})

    encoding_table = encoding_table.fillna(0)

    encoding_table.columns = [prefix+'_avg_cnt_month', prefix+'_avg_revenue_month',prefix+'_avg_price_month']

    encoding_table = encoding_table.reset_index()

    

    encoding_table[['date_block_num',prefix+'_avg_cnt_month']] = encoding_table[['date_block_num',prefix+'_avg_cnt_month']].astype('float16')

    encoding_table[encode_item] = encoding_table[encode_item].astype('int16')

    encoding_table[[prefix+'_avg_revenue_month',prefix+'_avg_price_month']] = encoding_table[[prefix+'_avg_revenue_month',prefix+'_avg_price_month']].astype('float32')

    

    return encoding_table
def first_sale_cnt(dataset,source_set,keys,prefix,forward_range=6):

    """

    function to calculate avg. first sale count

    

    Parameters: 

    ------------

    dataset : (pd.DataFrame) target dataset 

    

    source_set :(pd.DataFrame) source dataframe to judge first/last and avg. sale count

    

    keys : (list) level of group. ex. ['shop_id','item_id'] means judging by shop-item pair 

    

    prefix : (str) prefix of created columns

    

    forward_range : (int) include how many months data to calculate. It may have large variage by judging only one month, 

        because their would be only few items are been sold for the fisrt time in the last month

    ------------

    """    

    #filter unnecessary data (not first sale, extra columns)

    source_set = source_set[keys+['date_block_num','item_first_sold_offset','item_cnt_month']]

    source_set = source_set[source_set['item_first_sold_offset']==0]

    source_set = source_set.drop('item_first_sold_offset',axis=1)

    source_set = source_set[source_set['item_cnt_month']>0] #extra zeros are padded to simulate test set

    

    for m in range(1,35): #1~34m      

        

        #take previous 6 months data to prevent large variation

        tmp = source_set[source_set['date_block_num'].between(m-1-6,m-1)].groupby(keys).agg({'item_cnt_month':[np.mean]})

        tmp.columns = [prefix + '_avg_first_sold_cnt_month']

        tmp = tmp.reset_index()

        tmp['date_block_num'] = m #(m-1-6 to m-1)data is for month m

        

        if m == 1:  

            avg_first_sold_cnt = tmp

        else:

            avg_first_sold_cnt = pd.concat([avg_first_sold_cnt,tmp],axis=0)

            

        del tmp

        #breakpoint()

        

    dataset = dataset.merge(avg_first_sold_cnt,on=keys+['date_block_num'],how='left')

        

    columns = [prefix + '_avg_first_sold_cnt_month']

    dataset[columns] = dataset[columns].fillna(0)

    dataset[columns] = dataset[columns].astype('float16')

    

    return dataset
def get_google_trends(IPs, file_path, reference):

    """

    function to access Google Trends to fetch search popularity trend for each keywords

    

    Parameters: 

    ------------

    IPs : (list) list of keywords need to be searched

    

    file_path :(str) path to save/load Google Trend data since API has a usage limit

    

    reference : (str) a reference keyword to make Google Trend data can be compared across different keywords 

    ------------

    """    

    

    pytrend = TrendReq(hl='ru', tz=360)

    

    try:

        IP_trend = pd.read_csv(file_path)

        existed_columns = IP_trend.columns

    except FileNotFoundError:

        IP_trend = None

        existed_columns = []

    #breakpoint()  

    for IP in IPs:

        if IP in existed_columns:

            continue

        

        #every trend is normalized to itself, so a general reference will make it comparable with others

        if IP == reference:

            keywords = [reference]

        else:

            keywords = [IP,reference]

        pytrend.build_payload(kw_list=keywords,cat=0,timeframe='2013-01-01 2015-10-31',geo='RU',gprop='')



        trend_df = pytrend.interest_over_time()

        trend_df = trend_df.drop(reference,axis=1)

        if len(trend_df) == 0:

            #no key word trends found

            if IP_trend is not None:

                IP_trend[IP] = 0

        else:    

            trend_df = trend_df.reset_index()



            trend_df['year'] = trend_df['date'].dt.year

            trend_df['month'] = trend_df['date'].dt.month



            trend_df = trend_df.drop(['date','isPartial'],axis=1)

            trend_df = trend_df.groupby(['year','month']).max().reset_index()



            if IP_trend is None:

                IP_trend = trend_df

            else:       

                IP_trend = IP_trend.merge(trend_df,on=['year','month'],how='left')



# cannot override the file system on kaggle

#     IP_trend.to_csv(file_path, index=False)

    return IP_trend
%%time

lag_months = [1,2,3,6,12]

lag_features_list = ['item_cnt_month']

on_key = ['shop_id','item_id','date_block_num']

fillna = [0]

dtypes = ['int8']



dataset = add_lag_v3(dataset,None,lag_months,lag_features_list,on_key,fillna,dtypes)
%%time

lag_months = [1]

lag_features_list = ['avg_price']

on_key = ['shop_id','item_id','date_block_num']

fillna = [0]

dtypes = ['float32']



dataset = add_lag_v3(dataset,None,lag_months,lag_features_list,on_key,fillna,dtypes)
#add lag first sale month and last sale mont

dataset = add_first_last_sale(dataset,df,['shop_id','item_id'],'pair')

dataset = dataset.drop(['pair_first_sold','pair_last_sold'],axis=1)
%%time

item_month_sale = create_mean_encoding_table(dataset, ['item_id'], 'item')



lag_months = [1]

lag_features_list = ['item_avg_cnt_month', 'item_avg_revenue_month', 'item_avg_price_month']

on_key = ['item_id','date_block_num']

fillna = [0,0,0]

dtypes = ['float16','float32','float32']



dataset = add_lag_v3(dataset,item_month_sale,lag_months,lag_features_list,on_key,fillna,dtypes)



dataset = dataset.drop(['item_avg_cnt_month','item_avg_revenue_month', 'item_avg_price_month'],axis=1)

del item_month_sale

gc.collect()
%%time

#add lag first sale month and last sale mont

dataset = add_first_last_sale(dataset,df,['item_id'],'item')

dataset = dataset.drop(['item_first_sold','item_last_sold'],axis=1)
#merge item_category id into main dataset

dataset = pd.merge(dataset,items[['item_id','item_category_id']],how='left',on=['item_id'])

dataset['item_category_id'] = dataset['item_category_id'].astype('int16')
#google API has daily usage limit, try to read file first

try:

    item_categories_plus = pd.read_csv(tables_path + 'item_categories_plus.csv')

except FileNotFoundError:

    item_categories_plus = item_categories.copy()

    #translation

    trans = Translator()

    item_categories_plus['item_category_name_en'] = item_categories_plus['item_category_name'].apply(lambda name: trans.translate(name,dest='en').text)

    item_categories_plus

    item_categories_plus.to_csv(tables_path + 'item_categories_plus.csv')

    

#roughly grouping first

item_categories_plus['main_category'] = item_categories_plus['item_category_name_en'].apply(lambda name: name.split('-')[0].strip())

item_categories_plus['sub_category'] = item_categories_plus['item_category_name_en'].apply(

    lambda name: (name.split('-')[1].strip()) if len(name.split('-'))>1 else (name.split('-')[0].strip()))



#manually correct grouping

item_categories_plus.loc[24,'main_category'] = 'Games' #translation failed

item_categories_plus.loc[71,'main_category'] = 'Platic Bags and Gift' #plastic bag outlier -> stand alone group

item_categories_plus.loc[78,'main_category'] = 'Program' #programs -> program

item_categories_plus.loc[32,'main_category'] = 'Payment cards' #Payment card (Movies, Music, Games) -> Payment cards, and (Movies, Music, Games)

item_categories_plus.loc[32,'sub_category'] = '(Movies, Music, Games)' #Payment card  -> Payment cards

item_categories_plus.loc[36,'main_category'] = 'Payment cards'

item_categories_plus.loc[37:41,'main_category'] = 'Movies' #correct Movie, Movies, Cinema two groups



#label encoding

main_category_encoder = LabelEncoder()

item_categories_plus['main_category_enc'] = main_category_encoder.fit_transform(item_categories_plus['main_category']).astype('int8')

sub_category_encoder = LabelEncoder()

item_categories_plus['sub_category_enc'] = sub_category_encoder.fit_transform(item_categories_plus['sub_category']).astype('int8')



dataset = pd.merge(dataset,item_categories_plus[['item_category_id','main_category_enc','sub_category_enc']],how='left',on=['item_category_id'])



del item_categories_plus

gc.collect()
# %%time

# cate_month_sale = create_mean_encoding_table(dataset, ['item_category_id'], 'cate')



# lag_months = [1]

# lag_features_list = ['cate_avg_cnt_month', 'cate_avg_revenue_month', 'cate_avg_price_month']

# on_key = ['item_category_id','date_block_num']

# fillna = [0,0,0]

# dtypes = ['float16','float32','float32']



# dataset = add_lag_v3(dataset,cate_month_sale,lag_months,lag_features_list,on_key,fillna,dtypes)

# dataset = dataset.drop(['cate_avg_cnt_month', 'cate_avg_revenue_month','cate_avg_price_month'],axis=1)



# del cate_month_sale

# gc.collect()
%%time

main_cate_month_sale = create_mean_encoding_table(dataset, ['main_category_enc'], 'main_cate')



lag_months = [1]

lag_features_list = ['main_cate_avg_cnt_month', 'main_cate_avg_revenue_month', 'main_cate_avg_price_month']

on_key = ['main_category_enc','date_block_num']

fillna = [0,0,0]

dtypes = ['float16','float32','float32']



dataset = add_lag_v3(dataset,main_cate_month_sale,lag_months,lag_features_list,on_key,fillna,dtypes)

dataset = dataset.drop(['main_cate_avg_cnt_month','main_cate_avg_revenue_month', 'main_cate_avg_price_month'],axis=1)



del main_cate_month_sale

gc.collect()
%%time

sub_cate_month_sale = create_mean_encoding_table(dataset, ['sub_category_enc'], 'sub_cate')



lag_months = [1]

lag_features_list = ['sub_cate_avg_cnt_month', 'sub_cate_avg_revenue_month', 'sub_cate_avg_price_month']

on_key = ['sub_category_enc','date_block_num']

fillna = [0,0,0]

dtypes = ['float16','float32','float32']



dataset = add_lag_v3(dataset,sub_cate_month_sale,lag_months,lag_features_list,on_key,fillna,dtypes)

dataset = dataset.drop(['sub_cate_avg_cnt_month','sub_cate_avg_revenue_month', 'sub_cate_avg_price_month'],axis=1)



del sub_cate_month_sale

gc.collect()
%%time

#add first sold information for categories

dataset = first_sale_cnt(dataset,dataset,['item_category_id'],'cate')

dataset = first_sale_cnt(dataset,dataset,['main_category_enc'],'main_cate')

dataset = first_sale_cnt(dataset,dataset,['sub_category_enc'],'sub_cate')
#parse year/month information from 'date'

date_features = df.drop_duplicates(['date_block_num'])[['date','date_block_num']]

date_features['year'] = date_features['date'].apply(lambda d: d.split('.')[2])

date_features['month'] = date_features['date'].apply(lambda d: d.split('.')[1])

date_features = date_features.drop('date',axis=1).reset_index(drop=True)



#fill year/month information for test data

test_date = [34,2015,11]

date_features.loc[34] = test_date



#merge into main dataset

dataset = pd.merge(dataset,date_features,how='left',on=['date_block_num'])



#compress and delete temporary dataset to release memory

dataset[['month','year']] = dataset[['month','year']].astype('int16')



#keep date_features for later usage
dataset['shop_id'] = dataset['shop_id'].replace({0:57,1:58,10:11})

df['shop_id'] = df['shop_id'].replace({0:57,1:58,10:11})
try:

    shops_plus = pd.read_csv(tables_path + 'shops_plus.csv')

except FileNotFoundError:

    shops_plus = shops.copy()

    trans = Translator()

    shops_plus['shop_name_en'] = shops_plus['shop_name'].apply(lambda name: trans.translate(name,dest='en').text)

    shops_plus.to_csv(tables_path + 'shops_plus.csv')



#extract city

shops_plus['shop_city'] = shops_plus['shop_name_en'].apply(lambda shop: shop.split(' ')[0].strip())



#manually correct

shops_plus.loc[0:1,'shop_city'] = 'Yakutsk' #correct name

shops_plus.loc[12,'shop_city'] = 'Online_emergency' #online emergency shop

shops_plus.loc[20,'shop_city'] = 'Sale' #seperate from Moscow

shops_plus.loc[42:43,'shop_city'] = 'St.Petersburg' #correct name



#encode

shop_enc = LabelEncoder()

shops_plus['shop_city_enc'] = shop_enc.fit_transform(shops_plus['shop_city'])

shops_plus['shop_city_enc'] = shops_plus['shop_city_enc'].astype('int8')



#merge city index into main table

dataset = dataset.merge(shops_plus[['shop_id','shop_city_enc']],how='left',on=['shop_id'])

del shops_plus

gc.collect()
#add lag first sale month and last sale mont

dataset = add_first_last_sale(dataset,df,['shop_id'],'shop')

dataset = dataset.drop(['shop_first_sold','shop_last_sold'],axis=1)
%%time

shop_month_sale = create_mean_encoding_table(dataset, ['shop_id'], 'shop')



lag_months = [1,2,3,6,12]

lag_features_list = ['shop_avg_cnt_month', 'shop_avg_revenue_month', 'shop_avg_price_month']

on_key = ['shop_id','date_block_num']

fillna = [0,0,0]

dtypes = ['float16','float32','float32']



dataset = add_lag_v3(dataset,shop_month_sale,lag_months,lag_features_list,on_key,fillna,dtypes)

dataset = dataset.drop(['shop_avg_cnt_month', 'shop_avg_revenue_month','shop_avg_price_month'],axis=1)



del shop_month_sale

gc.collect()
%%time

shop_city_month_sale = create_mean_encoding_table(dataset, ['shop_city_enc'], 'shop_city')



lag_months = [1]

lag_features_list = ['shop_city_avg_cnt_month', 'shop_city_avg_revenue_month', 'shop_city_avg_price_month']

on_key = ['shop_city_enc','date_block_num']

fillna = [0,0,0]

dtypes = ['float16','float32','float32']



dataset = add_lag_v3(dataset,shop_city_month_sale,lag_months,lag_features_list,on_key,fillna,dtypes)

dataset = dataset.drop(['shop_city_avg_cnt_month','shop_city_avg_revenue_month','shop_city_avg_price_month'],axis=1)



del shop_city_month_sale

gc.collect()
# %%time

# shop_cate_month_sale = create_mean_encoding_table(dataset, ['shop_id','item_category_id'], 'shop_cate')



# lag_months = [1,2,3,6,12]

# lag_features_list = ['shop_cate_avg_cnt_month', 'shop_cate_avg_revenue_month', 'shop_cate_avg_price_month']

# on_key = ['shop_id','item_category_id','date_block_num']

# fillna = [0,0,0]

# dtypes = ['float16','float32','float32']



# dataset = add_lag_v3(dataset,shop_cate_month_sale,lag_months,lag_features_list,on_key,fillna,dtypes)

# dataset = dataset.drop(['shop_cate_avg_revenue_month','shop_cate_avg_revenue_month','shop_cate_avg_price_month'],axis=1)



# del shop_cate_month_sale

# gc.collect()
%%time

shop_main_cate_month_sale = create_mean_encoding_table(dataset, ['shop_id','main_category_enc'], 'shop_main_cate')



lag_months = [1]

lag_features_list = ['shop_main_cate_avg_cnt_month', 'shop_main_cate_avg_revenue_month', 'shop_main_cate_avg_price_month']

on_key = ['shop_id','main_category_enc','date_block_num']

fillna = [0,0,0]

dtypes = ['float16','float32','float32']



dataset = add_lag_v3(dataset,shop_main_cate_month_sale,lag_months,lag_features_list,on_key,fillna,dtypes)

dataset = dataset.drop(['shop_main_cate_avg_cnt_month','shop_main_cate_avg_revenue_month','shop_main_cate_avg_price_month'],axis=1)



del shop_main_cate_month_sale

gc.collect()
%%time

shop_sub_cate_month_sale = create_mean_encoding_table(dataset, ['shop_id','sub_category_enc'], 'shop_sub_cate')



lag_months = [1,2,3,6,12]

lag_features_list = ['shop_sub_cate_avg_cnt_month', 'shop_sub_cate_avg_revenue_month', 'shop_sub_cate_avg_price_month']

on_key = ['shop_id','sub_category_enc','date_block_num']

fillna = [0,0,0]

dtypes = ['float16','float32','float32']



dataset = add_lag_v3(dataset,shop_sub_cate_month_sale,lag_months,lag_features_list,on_key,fillna,dtypes)

dataset = dataset.drop(['shop_sub_cate_avg_cnt_month','shop_sub_cate_avg_revenue_month','shop_sub_cate_avg_price_month'],axis=1)



del shop_sub_cate_month_sale

gc.collect()
%%time

#extra table item_id - IPs (names or titles or certain series of games, movies, characters)

item_IP = pd.read_csv(items_IP_path) #manually groupped table

IP_label = LabelEncoder()

item_IP['IP_encode'] = IP_label.fit_transform(item_IP['Search_Key_Word'])

dataset = dataset.merge(item_IP[['item_id','IP_encode']],on='item_id',how='left')

dataset['IP_encode'] = dataset['IP_encode'].astype('int16')



#google trends API to get trends, (will not override existed data)

IPs = list(item_IP['Search_Key_Word'].unique())

IPs.remove('Others')

IP_trend = get_google_trends(IPs, items_IP_trend_path,'Total War') #collect google trends info by IP keywords



#reshape to long format from wide to be able to merge

date_features[['year','month']] = date_features[['year','month']].astype('int64')

IP_trend = IP_trend.merge(date_features,on=['year','month'],how='left')

IP_trend = IP_trend.drop(['month','year'],axis=1)

IP_trend = IP_trend.set_index('date_block_num').stack().reset_index()

IP_trend.columns = ['date_block_num','Search_Key_Word','google_trends']



#lookup IP_encode (this way allow csv to save old key words and their trends)

IP_trend = IP_trend.merge(item_IP[['Search_Key_Word','IP_encode']].drop_duplicates(),on='Search_Key_Word',how='left')

IP_trend = IP_trend.dropna()

IP_trend = IP_trend.sort_values(by=['IP_encode','date_block_num'])
%%time

lag_months = [1,2,3,6,12]

lag_features_list = ['google_trends']

on_key = ['IP_encode','date_block_num']

fillna = [0]

dtypes = ['int8']



dataset = add_lag_v3(dataset,IP_trend,lag_months,lag_features_list,on_key,fillna,dtypes)

dataset = dataset.drop(['Search_Key_Word','google_trends'],axis=1)

del IP_trend

gc.collect()
%%time

IP_month_sale = create_mean_encoding_table(dataset, ['IP_encode'], 'IP')



lag_months = [1]

lag_features_list = ['IP_avg_cnt_month', 'IP_avg_revenue_month', 'IP_avg_price_month']

on_key = ['IP_encode','date_block_num']

fillna = [0,0,0]

dtypes = ['float16','float32','float32']



dataset = add_lag_v3(dataset,IP_month_sale,lag_months,lag_features_list,on_key,fillna,dtypes)

dataset = dataset.drop(['IP_avg_cnt_month', 'IP_avg_revenue_month','IP_avg_price_month'],axis=1)

del IP_month_sale

gc.collect()
del IPs

del item_IP

del date_features



del lag_months

del lag_features_list

del on_key

del fillna

del dtypes



del IP_label

del LabelEncoder



del df

del df_test

del sample_submission

del items

del shops

gc.collect()
assert dataset.isnull().sum().sum() == 0

assert np.isfinite(dataset).sum().sum() == dataset.shape[0] * dataset.shape[1]



dataset = dataset.drop(['avg_price','revenue'],axis=1)



dataset.to_pickle(processed_data_path + 'dataset.pkl')



# del dataset

# gc.collect()
dataset = pd.read_pickle(processed_data_path + 'dataset.pkl')
dataset.columns
#feature selection

selected = [

    'item_cnt_month', #must (target)

    'date_block_num', #must (to split training/validation/testing, will be omitted later)

    

    #pairs-related

    'item_cnt_month_lag_1m','item_cnt_month_lag_2m','item_cnt_month_lag_3m',

    'item_cnt_month_lag_6m','item_cnt_month_lag_12m','pair_first_sold_offset',

    'avg_price_lag_1m',

    

    #items-related

    'item_avg_cnt_month_lag_1m','item_first_sold_offset','item_avg_price_month_lag_1m',

    

    #item categories-related

    'main_cate_avg_cnt_month_lag_1m','main_cate_avg_revenue_month_lag_1m',

    'sub_cate_avg_revenue_month_lag_1m','sub_cate_avg_cnt_month_lag_1m',    

    'cate_avg_first_sold_cnt_month','main_cate_avg_first_sold_cnt_month','sub_cate_avg_first_sold_cnt_month',

    'main_cate_avg_price_month_lag_1m','sub_cate_avg_price_month_lag_1m',

    

    #date

    'month',

    

    #shops-related

    'shop_avg_cnt_month_lag_1m','shop_avg_cnt_month_lag_2m','shop_avg_cnt_month_lag_3m',

    'shop_avg_cnt_month_lag_6m','shop_avg_cnt_month_lag_12m','shop_city_avg_cnt_month_lag_1m',

    'shop_avg_price_month_lag_1m','shop_first_sold_offset','shop_last_sold_offset',

    'shop_main_cate_avg_cnt_month_lag_1m','shop_sub_cate_avg_cnt_month_lag_1m','shop_sub_cate_avg_cnt_month_lag_2m',

    'shop_sub_cate_avg_cnt_month_lag_3m','shop_sub_cate_avg_cnt_month_lag_6m','shop_sub_cate_avg_cnt_month_lag_12m',

    

    #IPs and google trends

    'IP_avg_cnt_month_lag_1m','IP_avg_revenue_month_lag_1m',

    'google_trends_lag_1m','google_trends_lag_2m','google_trends_lag_3m',

    'google_trends_lag_6m','google_trends_lag_12m',

]

dataset = dataset[selected]
#exclude first year from training because incomplete lagged features, and take month 33 as validation , monthe 34 as testing

train = dataset[dataset['date_block_num'].between(12,32)]

validate = dataset[dataset['date_block_num']==33]

test = dataset[dataset['date_block_num']==34]



del dataset

gc.collect()
#check the monthes

print(train['date_block_num'].unique())

print(validate['date_block_num'].unique())

print(test['date_block_num'].unique())
#X,y split

X_train = train.drop(['item_cnt_month','date_block_num'],axis=1)

y_train = train['item_cnt_month']



X_validate = validate.drop(['item_cnt_month','date_block_num'],axis=1)

y_validate = validate['item_cnt_month']



X_test = test.drop(['item_cnt_month','date_block_num'],axis=1)



del train

del test

del validate

gc.collect()
#features for model training

X_train.columns
#final check of shape and missing

print(f'training shape: X {X_train.shape}, y {y_train.shape}')

print(f'validation shape: X {X_validate.shape}, y {y_validate.shape}')

print(f'testing shape: X {X_test.shape}')



print(f'training missing: X {X_train.isnull().sum().sum()}, y {y_train.isnull().sum().sum()}')

print(f'validation missing: X {X_validate.isnull().sum().sum()}, y {y_validate.isnull().sum().sum()}')

print(f'testing missing: X {X_test.isnull().sum().sum()}')
#able to save multiple model hyperparameters and select one by comment out / uncomment

def create_model():

    

    model = xgb.XGBRegressor(eta=0.01, max_depth=10,n_estimators=2000,

                             colsample_bytree=0.5,subsample=0.8,

                             gamma=2, reg_alpha=0, reg_lambda=2, min_child_weight=300,

                             tree_method='gpu_hist', gpu_id=0, max_bin=2048,

                             n_jobs=-1)

    return model
%%time

model = create_model()

model.fit(X_train,y_train,eval_set=[(X_train,y_train),(X_validate,y_validate)],eval_metric='rmse',early_stopping_rounds=10)
with open( model_path + 'model.pkl', 'wb') as f:

    pickle.dump(model, f)
fig = plt.figure(figsize=(8,24))

ax1 = fig.add_subplot(311)

xgb.plot_importance(model,ax=ax1)



ax2 = fig.add_subplot(312)

xgb.plot_importance(model,ax=ax2,importance_type='gain',title='Gain')



ax3 = fig.add_subplot(313)

xgb.plot_importance(model,ax=ax3,importance_type='cover',title='Cover')
pred_val = model.predict(X_validate)

residue = y_validate-pred_val



plt.figure(figsize=(6,4))

g = sns.distplot(residue,bins=35,kde=False)

g.set(yscale='log')

g.set(xlabel='residue', ylabel='frequency')
pred_test = model.predict(X_test)

output = pd.DataFrame({'item_cnt_month':pred_test})

output.index.name = 'ID'
output = output.clip(0,20)

output.to_csv('./submission.csv')