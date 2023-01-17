!ls -sSh ../input/
import datetime

import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
import numpy as np
import pandas as pd
import seaborn as sns
from lightfm import LightFM
from lightfm.evaluation import auc_score
from scipy.sparse import coo_matrix
from sklearn import preprocessing

sns.set()
events = pd.read_csv('../input/events.csv')
print('Shape:', events.shape)
print('Columns', events.columns.tolist())
events.head()
data = events.event.value_counts()
labels = data.index
sizes = data.values
explode = (0, 0.1, 0.2)
fig, ax = plt.subplots(figsize=(6,6))
colors = ['b', 'g', 'r']

patches, texts, autotexts = ax.pie(sizes, labels=labels, explode=explode, autopct='%1.2f%%', shadow=False, startangle=90, colors=colors)

properties = fm.FontProperties()
properties.set_size('x-large')
# font size include: xx-small, x-small, small, medium, large, x-large, xx-large, larger, smaller, None

plt.setp(autotexts, fontproperties=properties)
plt.setp(texts, fontproperties=properties)

ax.axis('equal')
plt.show()
items = events.itemid.value_counts()
for i in [2, 10, 50, 100, 1000]:
    print(f'Items that appear less than {i:>4} times:\
          {round((items < i).mean() * 100, 2)}%')

items.head(10)
plt.figure(figsize=(16, 9))
plt.hist(items.values, bins=50, log=True)
plt.xlabel('Number of times item appeared', fontsize=16)
plt.ylabel('log(Count of displays with item)', fontsize=16)
plt.show()
visitors = events.visitorid.value_counts()
for i in [2, 5, 10, 15]:
    print(f'Visitors that appear less than {i:>2} times:\
          {round((visitors < i).mean() * 100, 2):.02f}%')

visitors.head(10)
plt.figure(figsize=(16, 9))
plt.hist(visitors.values, bins=50, log=True)
plt.xlabel('Number of times visitor appeared', fontsize=16)
plt.ylabel('log(Count of displays with visitor)', fontsize=16)
plt.show()
hour = lambda x: (datetime.datetime.fromtimestamp(x)-datetime.timedelta(hours=5)).hour
timestamp = events[['timestamp', 'event']].copy()
timestamp['timestamp'] = timestamp.timestamp / 1000
timestamp['hour'] = timestamp['timestamp'].apply(hour)

timestamp.head()
plt.figure(figsize=(12,6))
timestamp.hour.hist(bins=np.linspace(-0.5, 23.5, 25), alpha=1, density=True)
plt.xlim(-0.5, 23.5)
plt.xlabel("Hour of Day")
plt.ylabel("Fraction of Events")
plt.show()
properties = pd.concat([pd.read_csv('../input/item_properties_part1.csv'), pd.read_csv('../input/item_properties_part2.csv')])
print('Shape:', properties.shape)
print('Columns', properties.columns.tolist())
properties.head()
properties = properties.loc[properties.property.isin(['categoryid', 'available']), :]
print('Shape:', properties.shape)
properties.head()
categoryid = properties[properties.property=='categoryid'].drop_duplicates('itemid', keep='first')
available = properties[properties.property=='available']
categoryid.head()
categories = categoryid.value.value_counts()
categories.head(10)
for i in [2, 10, 50, 100, 500, 1000, 5000]:
    print(f'Categories that appear less than {i:>4} times:\
          {round((categories < i).mean() * 100, 2)}%')
plt.figure(figsize=(16, 9))
plt.hist(categories.values, bins=50, log=True)
plt.xlabel('Number of times categories appeared', fontsize=16)
plt.ylabel('log(Count of displays with category)', fontsize=16)
plt.show()
item_category = categoryid[['itemid', 'value']]
item_category.columns = ['itemid', 'categoryid']
item_category.head()
available.head()
category = pd.read_csv('../input/category_tree.csv').dropna()
print('Shape:', category.shape)
print('Columns', category.columns.tolist())
category.head()
category_parent_dict = category.set_index('categoryid').T.to_dict('list')

pd.options.mode.chained_assignment = None
item_category['parentid'] = item_category.categoryid.apply(lambda x: int(category_parent_dict.get(int(x), [x])[0]))

item_category.head()
# Format the timestamp as a date and arrange it in chronological order.
# 将时间戳格式化为日期，并且按时间顺序排列。
events = events.assign(date=pd.Series(datetime.datetime.fromtimestamp(i/1000).date() for i in events.timestamp))
events = events.sort_values('date').reset_index(drop=True)
events = events[['visitorid','itemid','event', 'date']]
events.head()
events.tail()
start_date = '2015-5-3'
end_date = '2015-5-18'
fd = lambda x: datetime.datetime.strptime(x, '%Y-%m-%d').date()
events = events[(events.date >= fd(start_date)) & (events.date <= fd(end_date))]
# Divide training sets and test sets
# 划分训练集和测试集
split_point = np.int(np.round(events.shape[0]*0.8))
events_train = events.iloc[0:split_point]
events_test = events.iloc[split_point::]
events_test = events_test[(events_test['visitorid'].isin(events_train['visitorid'])) & (events_test['itemid'].isin(events_train['itemid']))]
id_cols=['visitorid','itemid']
trans_cat_train=dict()
trans_cat_test=dict()

for k in id_cols:
    cate_enc=preprocessing.LabelEncoder()
    trans_cat_train[k]=cate_enc.fit_transform(events_train[k].values)
    trans_cat_test[k]=cate_enc.transform(events_test[k].values)
ratings = dict()

cate_enc=preprocessing.LabelEncoder()
ratings['train'] = cate_enc.fit_transform(events_train.event)
ratings['test'] = cate_enc.transform(events_test.event)
n_users=len(np.unique(trans_cat_train['visitorid']))
n_items=len(np.unique(trans_cat_train['itemid']))
rate_matrix = dict()
rate_matrix['train'] = coo_matrix((ratings['train'], (trans_cat_train['visitorid'], trans_cat_train['itemid'])), shape=(n_users,n_items))
rate_matrix['test'] = coo_matrix((ratings['test'], (trans_cat_test['visitorid'], trans_cat_test['itemid'])), shape=(n_users,n_items))
model = LightFM(no_components=5, loss='warp')
model.fit(rate_matrix['train'], epochs=100, num_threads=8)
auc_score(model, rate_matrix['train'], num_threads=8).mean()
auc_score(model, rate_matrix['test'], num_threads=8).mean()