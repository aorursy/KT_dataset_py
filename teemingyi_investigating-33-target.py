import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import gc
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data_train = pd.read_csv('../input/elo-merchant-category-recommendation/train.csv')
data_test = pd.read_csv('../input/elo-merchant-category-recommendation/test.csv')
data_train.target.value_counts().sort_index()
data_train.first_active_month.value_counts().sort_index().plot()
data_test.first_active_month.value_counts().sort_index().plot()
percent_low = data_train.target.value_counts(normalize=True).sort_index().iloc[0]
print('We expect {:,.10} of -33.219 in the test set.'.format(percent_low * len(data_test.index)))
data_normal = data_train.loc[data_train.target>-30]
data_less = data_train.loc[data_train.target<-30]
pd.concat([data_normal.first_active_month.value_counts(normalize=True).sort_index(),
           data_less.first_active_month.value_counts(normalize=True).sort_index()],axis=1).plot()
plt.legend(['normal','-33'])

pd.concat([data_normal.feature_1.value_counts(normalize=True).sort_index(),
           data_less.feature_1.value_counts(normalize=True).sort_index()],axis=1).plot(kind='bar')
plt.legend(['normal','-33'])
pd.concat([data_normal.feature_2.value_counts(normalize=True).sort_index(),
           data_less.feature_2.value_counts(normalize=True).sort_index()],axis=1).plot(kind='bar')
plt.legend(['normal','-33'])
pd.concat([data_normal.feature_3.value_counts(normalize=True).sort_index(),
           data_less.feature_3.value_counts(normalize=True).sort_index()],axis=1).plot(kind='bar')
plt.legend(['normal','-33'])
card_id = list(data_train.card_id.unique()) + list(data_test.card_id.unique())
merchant_id = list(pd.read_csv('../input/elo-merchant-category-recommendation/merchants.csv',usecols=['merchant_id']).merchant_id.unique())
card_id_dict = {value: key for key, value in enumerate(card_id)}
merchant_id_dict = {value: key for key, value in enumerate(merchant_id)}
data_normal['card_id_label'] = data_normal.card_id.map(lambda x: card_id_dict[x])
data_less['card_id_label'] = data_less.card_id.map(lambda x: card_id_dict[x])
data_hist = pd.read_csv('../input/cleaning-historical-txns/historical_transactions.csv')
# Memory saving function credit to https://www.kaggle.com/gemartin/load-data-reduce-memory-usage
def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.        
    """
    #start_mem = df.memory_usage().sum() / 1024**2
    #print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    for col in df.columns:
        col_type = df[col].dtype

        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

    #end_mem = df.memory_usage().sum() / 1024**2
    #print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    #print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

    return df
data_hist.info()
data_hist.category_2 = data_hist.category_2.astype(int)
data_hist = reduce_mem_usage(data_hist)
data_hist.purchase_amount = data_hist.purchase_amount.astype(float)
gc.collect()
data_hist.info()
data_normal_hist = data_hist.loc[data_hist.card_id_label.isin(data_normal.card_id_label)]
data_less_hist = data_hist.loc[data_hist.card_id_label.isin(data_less.card_id_label)]
print('''There are {:,} transactions from the normal target card,
{:,} transactions from the -33 target card,
remaining {:,} transactions for the test set.'''.format(
len(data_normal_hist), len(data_less_hist), len(data_hist)-len(data_normal_hist)-len(data_less_hist)))
from scipy.stats import percentileofscore
authorised_normal = data_normal_hist.groupby('card_id_label').authorized_flag.agg(['count','mean'])
authorised_normal.head(10)
authorised_normal['count_less_than100'] = authorised_normal['count'].map(lambda x: 1 if x<100 else 0)
authorised_normal['have_unauthorised'] = authorised_normal['mean'].map(lambda x: 1 if x<1 else 0)
authorised_less = data_less_hist.groupby('card_id_label').authorized_flag.agg(['count','mean'])
authorised_less['count_less_than100'] = authorised_less['count'].map(lambda x: 1 if x<100 else 0)
authorised_less['have_unauthorised'] = authorised_less['mean'].map(lambda x: 1 if x<1 else 0)
pd.concat([authorised_normal.count_less_than100.value_counts(normalize=True).sort_index(),
           authorised_less.count_less_than100.value_counts(normalize=True).sort_index()],axis=1).plot(kind='bar')
plt.legend(['normal','-33'])
pd.concat([authorised_normal.have_unauthorised.value_counts(normalize=True).sort_index(),
           authorised_less.have_unauthorised.value_counts(normalize=True).sort_index()],axis=1).plot(kind='bar')
plt.legend(['normal','-33'])
data_normal_hist.groupby('card_id_label').authorized_flag.agg(['count']).describe()
data_less_hist.groupby('card_id_label').authorized_flag.agg(['count']).describe()
data_normal_hist.groupby('card_id_label').authorized_flag.agg(['count'])['count'].value_counts(normalize=True).sort_index().plot()
data_less_hist.groupby('card_id_label').authorized_flag.agg(['count'])['count'].value_counts(normalize=True).sort_index().plot()
data_normal_hist.groupby('card_id_label').authorized_flag.agg(['mean'])['mean'].describe()
data_less_hist.groupby('card_id_label').authorized_flag.agg(['mean'])['mean'].describe()
data_normal_hist.groupby('card_id_label').authorized_flag.agg(['mean'])['mean'].value_counts(normalize=True).sort_index().plot()
data_less_hist.groupby('card_id_label').authorized_flag.agg(['mean'])['mean'].value_counts(normalize=True).sort_index().plot()
def get_agg_stats(columns,func):
    normal = data_normal_hist.groupby('card_id_label')[columns].agg(func)
    
    less = data_less_hist.groupby('card_id_label')[columns].agg(func)
    return normal, less
cat1_normal, cat1_less = get_agg_stats('category_1',['mean'])
def add_new_columns(normal, less, new_column_name, column_name, func):
    normal[new_column_name] = normal[column_name].map(func)
    less[new_column_name] = less[column_name].map(func)
    return normal, less
cat1_normal, cat1_less = add_new_columns(cat1_normal, cat1_less, 'mixed', 'mean', lambda x: 1 if x>0 and x<1 else 0)
pd.concat([cat1_normal['mixed'].value_counts(normalize=True).sort_index(),
           cat1_less['mixed'].value_counts(normalize=True).sort_index()],axis=1).plot(kind='bar')
plt.legend(['normal','-33'])
instal_normal, instal_less = get_agg_stats('installments',['mean'])
instal_normal, instal_less = add_new_columns(instal_normal, instal_less, 'use_instal', 'mean', lambda x: 1 if x>0 else 0)
pd.concat([instal_normal['use_instal'].value_counts(normalize=True).sort_index(),
           instal_less['use_instal'].value_counts(normalize=True).sort_index()],axis=1).plot(kind='bar')
plt.legend(['normal','-33'])
cat2_normal, cat2_less = get_agg_stats('category_2',['nunique'])
cat2_normal, cat2_less = add_new_columns(cat2_normal, cat2_less, 'more_than_1', 'nunique', lambda x: 1 if x>1 else 0)
pd.concat([cat2_normal['more_than_1'].value_counts(normalize=True).sort_index(),
           cat2_less['more_than_1'].value_counts(normalize=True).sort_index()],axis=1).plot(kind='bar')
plt.legend(['normal','-33'])
cat3_normal, cat3_less = get_agg_stats('category_3',['nunique'])
cat3_normal, cat3_less = add_new_columns(cat3_normal, cat3_less, 'more_than_1', 'nunique', lambda x: 1 if x>1 else 0)
pd.concat([cat3_normal['more_than_1'].value_counts(normalize=True).sort_index(),
           cat3_less['more_than_1'].value_counts(normalize=True).sort_index()],axis=1).plot(kind='bar')
plt.legend(['normal','-33'])
merc_cat_normal, merc_cat_less = get_agg_stats('merchant_category_id',['nunique'])
merc_cat_normal, merc_cat_less = add_new_columns(merc_cat_normal, merc_cat_less, 'more_than_25', 'nunique', lambda x: 1 if x>25 else 0)
pd.concat([merc_cat_normal['more_than_25'].value_counts(normalize=True).sort_index(),
           merc_cat_less['more_than_25'].value_counts(normalize=True).sort_index()],axis=1).plot(kind='bar')
plt.legend(['normal','-33'])
month_lag_normal, month_lag_less = get_agg_stats('month_lag',['median'])
month_lag_normal, month_lag_less = add_new_columns(month_lag_normal, month_lag_less, 'more_than_-3', 'median', lambda x: 1 if x>-3 else 0)
pd.concat([month_lag_normal['more_than_-3'].value_counts(normalize=True).sort_index(),
           month_lag_less['more_than_-3'].value_counts(normalize=True).sort_index()],axis=1).plot(kind='bar')
plt.legend(['normal','-33'])
data_less_hist.columns
purchase_amount_normal = data_normal_hist.groupby('card_id_label').purchase_amount.agg(['sum','mean','median'])
purchase_amount_normal['sum_positive'] = purchase_amount_normal['sum'].map(lambda x: 1 if x>0 else 0)
purchase_amount_normal['mean_positive'] = purchase_amount_normal['mean'].map(lambda x: 1 if x>0 else 0)
purchase_amount_less = data_less_hist.groupby('card_id_label').purchase_amount.agg(['sum','mean','median'])
purchase_amount_less['sum_positive'] = purchase_amount_less['sum'].map(lambda x: 1 if x>0 else 0)
purchase_amount_less['mean_positive'] = purchase_amount_less['mean'].map(lambda x: 1 if x>0 else 0)
pd.concat([purchase_amount_normal.sum_positive.value_counts(normalize=True).sort_index(),
           purchase_amount_less.sum_positive.value_counts(normalize=True).sort_index()],axis=1).plot(kind='bar')
plt.legend(['normal','-33'])
pd.concat([purchase_amount_normal.mean_positive.value_counts(normalize=True).sort_index(),
           purchase_amount_less.mean_positive.value_counts(normalize=True).sort_index()],axis=1).plot(kind='bar')
plt.legend(['normal','-33'])
