import pandas as pd
orders = pd.read_csv('../input/orders.csv' )
products = pd.read_csv('../input/products.csv')
order_products = pd.read_csv('../input/order_products__train.csv' )
order_products_prior = pd.read_csv('../input/order_products__prior.csv')
aisles = pd.read_csv('../input/aisles.csv')
departments = pd.read_csv('../input/departments.csv')
order_tbl = orders
order_tbl.sort_values(['user_id', 'order_number'], inplace=True)
order_tbl['t-1_order_id'] = order_tbl.groupby('user_id')['order_id'].shift(1)
order_tbl['t-2_order_id'] = order_tbl.groupby('user_id')['order_id'].shift(2)
order_tbl['t-3_order_id'] = order_tbl.groupby('user_id')['order_id'].shift(3)
order_tbl.head()
col = ['order_id', 'order_dow', 'order_hour_of_day']
order_tbl = pd.merge(order_tbl, order_tbl[col].add_prefix('t-1_'), on='t-1_order_id', how='left')
order_tbl = pd.merge(order_tbl, order_tbl[col].add_prefix('t-2_'), on='t-2_order_id', how='left')
order_tbl = pd.merge(order_tbl, order_tbl[col].add_prefix('t-3_'), on='t-3_order_id', how='left')
order_tbl.head()
order_tbl['delta_hour_t-1'] = order_tbl['order_hour_of_day'] - order_tbl['t-1_order_hour_of_day']
order_tbl['delta_hour_t-2'] = order_tbl['order_hour_of_day'] - order_tbl['t-2_order_hour_of_day']
order_tbl['delta_hour_t-3'] = order_tbl['order_hour_of_day'] - order_tbl['t-3_order_hour_of_day']
order_tbl.head()
orders.head()
order_products_prior.head()
prd = pd.merge(orders, order_products_prior, on='order_id', how='inner')
prd.head(10)
import numpy as np
prd['user_max_onb'] = prd.groupby('user_id').order_number.transform(np.max)
prd = prd.groupby(['user_id', 'product_id']).head(2)
prd.head(30)
from collections import defaultdict
item_cnt    = defaultdict(int)
item_chance = defaultdict(int)
pid_bk = uid_bk = onb_bk = None
for uid, pid, onb, max_onb in prd[['user_id', 'product_id', 'order_number', 'user_max_onb']].values:
        
    if uid==uid_bk and pid==pid_bk and (onb-onb_bk==1):
        item_cnt[pid] +=1
    if onb!=max_onb:
        item_chance[pid] +=1
    
    pid_bk = pid
    uid_bk = uid
    onb_bk = onb
item_cnt = pd.DataFrame.from_dict(item_cnt, orient='index').reset_index()
item_cnt.columns = ['product_id', 'item_first_cnt']
item_chance = pd.DataFrame.from_dict(item_chance, orient='index').reset_index()
item_chance.columns = ['product_id', 'item_first_chance']
df = pd.merge(item_cnt, item_chance, on='product_id', how='outer').fillna(0)
df['item_first_ratio'] = df.item_first_cnt/df.item_first_chance
df.head()
user = prd.drop_duplicates('user_id')[['user_id']].reset_index(drop=True)
user.head()
tag_user = prd[prd.product_id==24852].user_id
user['hyb_Banana'] = 0
user.loc[user.user_id.isin(tag_user), 'hyb_Banana'] = 1
    
tag_user = prd[prd.product_id==13176].user_id
user['hyb_BoO-Bananas'] = 0
user.loc[user.user_id.isin(tag_user), 'hyb_BoO-Bananas'] = 1
    
tag_user = prd[prd.product_id==21137].user_id
user['hyb_Organic-Strawberries'] = 0
user.loc[user.user_id.isin(tag_user), 'hyb_Organic-Strawberries'] = 1
    
tag_user = prd[prd.product_id==21903].user_id
user['hyb_Organic-Baby-Spinach'] = 0
user.loc[user.user_id.isin(tag_user), 'hyb_Organic-Baby-Spinach'] = 1

tag_user = prd[prd.product_id==47209].user_id
user['hyb_Organic-Hass-Avocado'] = 0
user.loc[user.user_id.isin(tag_user), 'hyb_Organic-Hass-Avocado'] = 1
user.head()
prd['user_max_onb'] = prd.groupby('user_id').order_number.transform(np.max)   
item_N2_cnt    = defaultdict(int)
item_N2_chance = defaultdict(int)
item_N3_cnt    = defaultdict(int)
item_N3_chance = defaultdict(int)
item_N4_cnt    = defaultdict(int)
item_N4_chance = defaultdict(int)
item_N5_cnt    = defaultdict(int)
item_N5_chance = defaultdict(int)
pid_bk = uid_bk = onb_bk = None
for pid, uid, onb, max_onb in prd[['product_id', 'user_id', 'order_number','user_max_onb']].values:
        
    if pid==pid_bk and uid==uid_bk and (onb-onb_bk)<=2 and (max_onb-onb) >=2:
        item_N2_cnt[pid] +=1
    if pid==pid_bk and uid==uid_bk and (max_onb-onb) >=2:
        item_N2_chance[pid] +=1

    if pid==pid_bk and uid==uid_bk and (onb-onb_bk)<=3 and (max_onb-onb) >=3:
        item_N3_cnt[pid] +=1
    if pid==pid_bk and uid==uid_bk and (max_onb-onb) >=3:
        item_N3_chance[pid] +=1

    if pid==pid_bk and uid==uid_bk and (onb-onb_bk)<=4 and (max_onb-onb) >=4:
        item_N4_cnt[pid] +=1
    if pid==pid_bk and uid==uid_bk and (max_onb-onb) >=4:
        item_N4_chance[pid] +=1

    if pid==pid_bk and uid==uid_bk and (onb-onb_bk)<=5 and (max_onb-onb) >=5:
        item_N5_cnt[pid] +=1
    if pid==pid_bk and uid==uid_bk and (max_onb-onb) >=5:
        item_N5_chance[pid] +=1

    pid_bk = pid
    uid_bk = uid
    onb_bk = onb
item_N2_cnt = pd.DataFrame.from_dict(item_N2_cnt, orient='index').reset_index()
item_N2_cnt.columns = ['product_id', 'item_N2_cnt']
item_N2_chance = pd.DataFrame.from_dict(item_N2_chance, orient='index').reset_index()
item_N2_chance.columns = ['product_id', 'item_N2_chance']

item_N3_cnt = pd.DataFrame.from_dict(item_N3_cnt, orient='index').reset_index()
item_N3_cnt.columns = ['product_id', 'item_N3_cnt']
item_N3_chance = pd.DataFrame.from_dict(item_N3_chance, orient='index').reset_index()
item_N3_chance.columns = ['product_id', 'item_N3_chance']

item_N4_cnt = pd.DataFrame.from_dict(item_N4_cnt, orient='index').reset_index()
item_N4_cnt.columns = ['product_id', 'item_N4_cnt']
item_N4_chance = pd.DataFrame.from_dict(item_N4_chance, orient='index').reset_index()
item_N4_chance.columns = ['product_id', 'item_N4_chance']

item_N5_cnt = pd.DataFrame.from_dict(item_N5_cnt, orient='index').reset_index()
item_N5_cnt.columns = ['product_id', 'item_N5_cnt']
item_N5_chance = pd.DataFrame.from_dict(item_N5_chance, orient='index').reset_index()
item_N5_chance.columns = ['product_id', 'item_N5_chance']
df2 = pd.merge(item_N2_cnt, item_N2_chance, on='product_id', how='outer')
df3 = pd.merge(item_N3_cnt, item_N3_chance, on='product_id', how='outer')
df4 = pd.merge(item_N4_cnt, item_N4_chance, on='product_id', how='outer')
df5 = pd.merge(item_N5_cnt, item_N5_chance, on='product_id', how='outer')
df = pd.merge(pd.merge(df2, df3, on='product_id', how='outer'),
              pd.merge(df4, df5, on='product_id', how='outer'), 
              on='product_id', how='outer').fillna(0)
df['item_N2_ratio'] = df['item_N2_cnt']/df['item_N2_chance']
df['item_N3_ratio'] = df['item_N3_cnt']/df['item_N3_chance']
df['item_N4_ratio'] = df['item_N4_cnt']/df['item_N4_chance']
df['item_N5_ratio'] = df['item_N5_cnt']/df['item_N5_chance']
df.fillna(0, inplace=True)
df.reset_index(drop=True, inplace=True)
df.head(20)