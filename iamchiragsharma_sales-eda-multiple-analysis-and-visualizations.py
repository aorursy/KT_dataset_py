import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
pd.set_option('display.float_format', lambda x: '%.3f' % x)
!pip install pandas-profiling --quiet
!pip install googletrans --quiet
from googletrans import Translator
translator = Translator()
import glob
files = glob.glob("../input/competitive-data-science-predict-future-sales/*")
files = {file.split("/")[3][:-4] : file for file in files}
item_categories  = pd.read_csv(files['item_categories'])
item_categories.head()
items  = pd.read_csv(files['items'])
items.head()
#Takes a while to execute - Uncomment and execute if you want names in English otherwise move on.
# from tqdm import tqdm
# item_names = []
# for item in tqdm(items['item_name'].values):
#     item_names.append(translator.translate(item).text)
    
# items['item_name'] = item_names
# items.head()
sales_train = pd.read_csv(files['sales_train'])
sales_train.head()
shops = pd.read_csv(files['shops'])
shops.head()
shops['english_name'] = shops['shop_name'].apply(lambda shop_name : translator.translate(shop_name).text)
shops.head()
test = pd.read_csv(files['test'])
test.head()
sample_submission = pd.read_csv(files['sample_submission'])
sample_submission.head()
sales_train.head()
from pandas_profiling import ProfileReport
profile = ProfileReport(sales_train, title="Sales Data Profiling Report")
profile
sales_train['total_sale'] = sales_train.item_price * sales_train.item_cnt_day
sales_train.head()
months = ['Jan','Feb','Mar','Apr','May', 'Jun', 'Jul','Aug','Sept','Oct','Nov','Dec']
months_mapping = { i : str(months[i%12])+"'"+str(i//12 + 13) for i in range(34) }
sales_train['month-year'] = sales_train.date_block_num.map(months_mapping)
sales_train.head()
returned_items = sales_train[sales_train['item_cnt_day'] < 0]
returned_items['total_sale'] = returned_items['total_sale'] * -1
returned_items.head()
sales_train[sales_train['total_sale'] > 0].describe()
sales_train['date'] = pd.to_datetime(sales_train['date'])
sales_train.head()
total_sales = sales_train.groupby('date').sum()[['total_sale']]
total_sales.plot()
total_sales.groupby(pd.Grouper(freq='M')).sum().plot(figsize=(15,6))
sales_scatter = sales_train.loc[sales_train['total_sale'] > 0,['total_sale']]
sales_scatter['idx'] = sales_scatter.index
sns.scatterplot(x='idx',y='total_sale',data=sales_scatter)
plt.figure(figsize=(15,6))
sns.distplot(np.log(sales_scatter['total_sale']),kde=False)
sales_train.groupby('date').sum()[['total_sale']].sort_values('total_sale',ascending=False).head(30).plot(kind='bar',figsize=(15,6))
sales_train.groupby('date').sum()[['total_sale']].sort_values('total_sale',ascending=False).head(30).plot(kind='bar',figsize=(15,6),logy=True)
total_sales_by_date = sales_train.groupby('date').sum()[['total_sale']].sort_values('total_sale',ascending=False)
total_sales_by_date.head(15).append(total_sales_by_date.tail(15)).plot(kind='bar',figsize=(15,6))
total_sales_by_date = sales_train.groupby('date').sum()[['total_sale']].sort_values('total_sale',ascending=False)
total_sales_by_date.head(15).append(total_sales_by_date.tail(15)).plot(kind='bar',figsize=(15,6), logy=True)
total_items_by_date = sales_train.groupby('date').sum()[['item_cnt_day']].sort_values('item_cnt_day',ascending=False).head(30).plot(kind='bar', figsize=(15,6))
returned_items.groupby('date').count()[['item_cnt_day']].sort_values('item_cnt_day',ascending=False).head(30).plot(kind='bar',figsize=(15,6))
sales_train.groupby('month-year').sum()[['total_sale']].sort_values('total_sale',ascending=False).head(30).plot(kind='bar',figsize=(15,6))
returned_items.groupby('month-year').sum()[['total_sale']].sort_values('total_sale', ascending=False).head(30).plot(kind='bar',figsize=(15,6))
top5_month = returned_items.groupby(['month-year','item_id']).sum()[['total_sale']].sort_values('total_sale',ascending=False).iloc[:,:5]
top5_month.index.get_level_values(0).nunique()
plt.figure(figsize=(15,15))
sns.barplot(y='month-year',x='total_sale',data=top5_month.reset_index())
returned_items.groupby('item_id').count()[['date']].sort_values('date',ascending=False).head(20).plot(kind='bar', figsize=(15,6))
returned_items.groupby('date').count()[['item_cnt_day']].sort_values('item_cnt_day',ascending=False).head(20).plot(kind='bar', figsize=(15,6))
sales_train.groupby('shop_id').sum()[['total_sale']].sort_values('total_sale', ascending=False).head(30).plot(kind='bar', figsize=(15,6))
returned_items.groupby('shop_id').count()[['item_cnt_day']].sort_values('item_cnt_day', ascending=False).head(30).plot(kind='bar', figsize=(15,6))
returned_items.groupby('shop_id').sum()[['total_sale']].sort_values('total_sale', ascending=False).head(30).plot(kind='bar', figsize=(15,6))
shops_data = pd.merge(sales_train,shops)
shop_wise_revenue = shops_data.groupby('english_name').sum()[['total_sale']]
shop_wise_revenue['share'] = (shop_wise_revenue['total_sale']/shop_wise_revenue['total_sale'].sum())*100
top30_shops = shop_wise_revenue.sort_values('share', ascending=False).head(30)
shop_data_pie = top30_shops.to_dict()
shop_data_pie['total_sale']['Others'] = shop_wise_revenue['total_sale'].sum() - top30_shops['total_sale'].sum()
shop_data_pie['share']['Others'] = 100 - top30_shops['share'].sum()
shop_data_pie = pd.DataFrame(shop_data_pie)
shop_data_pie.tail()
fig, ax = plt.subplots(figsize=(10,10))
ax.pie(shop_data_pie['share'].values, labels=shop_data_pie.index.values, autopct='%1.1f%%',shadow=True, startangle=90)
ax.axis('equal')
ax.set_title("Revenue wise share of Top 30 shops and Others is remaning 30",pad=20.0, fontdict = {'fontsize': 20,'fontweight' : 'bold'})
plt.show()
item_data = pd.merge(sales_train,items)
item_share_revenue = item_data.groupby('item_name').sum()[['total_sale']].sort_values('total_sale',ascending=False)
item_share_revenue['share'] = (item_share_revenue['total_sale']/item_share_revenue['total_sale'].sum()) * 100
item_share_revenue.head()
top30_items = item_share_revenue.sort_values('share', ascending=False).head(20)
item_data_pie = top30_items.to_dict()
item_data_pie['total_sale']['Others'] = item_share_revenue['total_sale'].sum() - top30_items['total_sale'].sum()
item_data_pie['share']['Others'] = 100 - top30_items['share'].sum()
item_data_pie = pd.DataFrame(item_data_pie)
item_data_pie.head()
fig, ax = plt.subplots(figsize=(10,10))
ax.pie(item_data_pie['share'].values, labels=item_data_pie.index.values, autopct='%1.1f%%',shadow=True, startangle=90)
ax.axis('equal')
ax.set_title("Revenue wise share of Top 20 Items and Others",pad=20.0, fontdict = {'fontsize': 20,'fontweight' : 'bold'})
plt.show()


