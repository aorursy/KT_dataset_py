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
import pandas as pd

sales_train=pd.read_csv("/kaggle/input/competitive-data-science-predict-future-sales/sales_train.csv")

sample_sub=pd.read_csv("/kaggle/input/competitive-data-science-predict-future-sales/sample_submission.csv")

test=pd.read_csv("/kaggle/input/competitive-data-science-predict-future-sales/test.csv")

shop=pd.read_csv("/kaggle/input/competitive-data-science-predict-future-sales/shops.csv")

item=pd.read_csv("/kaggle/input/competitive-data-science-predict-future-sales/items.csv")

item_cat=pd.read_csv("/kaggle/input/competitive-data-science-predict-future-sales/item_categories.csv")

sample_sub
sales_train.sort_values(['date'])


import seaborn as sns

import numpy as np

sns.heatmap(sales_train.isnull(),yticklabels=False,cbar=False,cmap='viridis')
temp=sales_train[['item_id','item_price']]

item_dict = temp.set_index('item_id').T.to_dict('records')[0]

def get_value(x,item_dict):

    if x in item_dict.keys():

        return item_dict[x]

test['item_price']=test['item_id'].apply(get_value,item_dict=item_dict)
temp=item[['item_id','item_category_id']]

item_cat_dict = temp.set_index('item_id').T.to_dict('records')[0]

sales_train['item_category']=sales_train['item_id'].apply(get_value,item_dict=item_cat_dict)

test['item_category']=test['item_id'].apply(get_value,item_dict=item_cat_dict)

del temp
item_price_median=sales_train.groupby('item_category')['item_price'].median()
item_price_median=item_price_median.reset_index()

item_price_median
#replace item_price nulls with category median in test set

for i in range(len(item_price_median)):

    price=item_price_median['item_price'].iloc[i]

    cat=item_price_median['item_category'].iloc[i]

    test.loc[(test.item_price.isnull()) &(test.item_category==cat), 'item_price'] = price

sns.heatmap(test.isnull(),yticklabels=False,cbar=False,cmap='viridis')

del item_price_median
test
agg_df=sales_train.groupby(['date_block_num','shop_id','item_id']).agg({'item_cnt_day':'sum','item_price':"first",'item_category':'first'}).reset_index()
agg_df['month']=agg_df['date_block_num'].apply(lambda x:x%12)

agg_df=agg_df.rename(columns={'item_cnt_day':'item_cnt_month'})
item_df=agg_df.groupby('item_id').agg({'shop_id':pd.Series.nunique,'month':pd.Series.nunique}).reset_index()
item_df
from tqdm import tqdm

for i in tqdm(range(len(item_df))):

    agg_df.loc[agg_df.item_id==item_df.item_id.iloc[i],['no_of_month','no_of_shops']]=item_df['month'].iloc[i],item_df['shop_id'].iloc[i]

    test.loc[test.item_id==item_df.item_id.iloc[i],['no_of_month','no_of_shops']]=item_df['month'].iloc[i],item_df['shop_id'].iloc[i]
test['date_block_num']=[34 for _ in range(len(test))]

test['month']=[34%12 for _ in range(len(test))]
features=['date_block_num','shop_id','item_id','item_price','item_category']

target=['item_cnt_month']
fill_na_df=agg_df.groupby('item_category').agg({'no_of_month':'median','no_of_shops':'median'}).reset_index()
for i in range(len(fill_na_df)):

    month=fill_na_df['no_of_month'].iloc[i]

    shop=fill_na_df['no_of_shops'].iloc[i]

    cat=fill_na_df['item_category'].iloc[i]

    test.loc[(test.no_of_month.isnull()) &(test.item_category==cat), 'no_of_month'] = month

    test.loc[(test.no_of_shops.isnull()) &(test.item_category==cat), 'no_of_shops'] = shop

sns.heatmap(test.isnull(),yticklabels=False,cbar=False,cmap='viridis')

test.to_pickle("test.csv")

agg_df.to_pickle("agg_df.csv")