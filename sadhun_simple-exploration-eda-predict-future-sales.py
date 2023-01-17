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
import matplotlib.pyplot as plt
import seaborn as sns

color = sns.color_palette()

%matplotlib inline

items_df=pd.read_csv("/kaggle/input/competitive-data-science-predict-future-sales/items.csv")
shops_df=pd.read_csv("/kaggle/input/competitive-data-science-predict-future-sales/shops.csv")
sales_train_df=pd.read_csv("/kaggle/input/competitive-data-science-predict-future-sales/sales_train.csv")
test_df=pd.read_csv("/kaggle/input/competitive-data-science-predict-future-sales/test.csv")
#sample_df=pd.read_csv("/kaggle/input/competitive-data-science-predict-future-sales/sample_submission.csv")
item_categories_df=pd.read_csv("/kaggle/input/competitive-data-science-predict-future-sales/item_categories.csv")
items_df.head(3)
items_df.shape
shops_df.head(3)
item_categories_df.head(3)
sales_train_df.head(5)
sales_train_df.tail(5)
df_list = [items_df, sales_train_df, shops_df,item_categories_df,test_df]
df_list_names = ["items_df", "sales_train_df", "shops_df","item_categories_df","test_df"]

for i, n in zip(df_list, df_list_names):
    null_status = i.isnull().values.any()
    if null_status:
        print(n + " has null values")
    else:
        print(n + " doesn't have null values")
#from collections import defaultdict 
#item_cnt_dict=defaultdict()
#for key in sorted(sales_train_df.item_id.unique()):
#    item_cnt_dict[key]=[]
#for key in item_cnt_dict.keys():
#    item_cnt_dict[key]=sales_train_df[sales_train_df["item_id"]==key]["item_cnt_day"].sum()
#import operator
#N=5
#sorted_df = dict(sorted(item_cnt_dict.items(), key=operator.itemgetter(1),reverse=True)[:N])
#sorted_df
item_cnt_df=sales_train_df.groupby('item_id')['item_cnt_day'].sum().to_frame().reset_index().sort_values(by=['item_cnt_day'], ascending=False)[:10]
#item_cnt_df = pd.DataFrame(list(sorted_df.items()),columns = ['item_id','total_count']) 
merged_item_df = pd.merge(left=items_df, right=item_cnt_df, left_on='item_id', right_on='item_id')
merged_item_df = pd.merge(left=merged_item_df, right=item_categories_df, left_on='item_category_id', right_on='item_category_id')
merged_item_df=merged_item_df.sort_values('item_cnt_day',ascending=False)
merged_item_df

plt.figure(figsize=(12,8))

sns.barplot(x='item_id', y="item_cnt_day", data=merged_item_df,
            order=merged_item_df.sort_values('item_cnt_day',ascending = False).item_id,alpha=0.8,color=color[2])
plt.xlabel('Item_id', fontsize=12)
plt.ylabel('Total_count', fontsize=12)
plt.xticks(rotation='vertical')
plt.ylim(0,25000)
plt.show()
sales_train_df.head(3)
import operator
N=30
shop_cnt_dict={}
for key in np.sort(sales_train_df.shop_id.unique()).tolist():
    shop_cnt_dict[key]=[]
for key in shop_cnt_dict.keys():
    shop_cnt_dict[key]=sales_train_df[sales_train_df["shop_id"]==key]["item_cnt_day"].sum()
sorted_df = dict(sorted(shop_cnt_dict.items(), key=operator.itemgetter(1),reverse=True)[:N])
shop_cnt_df = pd.DataFrame(list(sorted_df.items()),columns = ['shop_id','total_count']) 
shop_cnt_df  

plt.figure(figsize=(12,8))

sns.barplot(x='shop_id', y="total_count", data=shop_cnt_df,
            order=shop_cnt_df.sort_values('total_count',ascending = False).shop_id,alpha=0.8,color=color[2])
plt.xlabel('Shop_id', fontsize=12)
plt.ylabel('Total_count', fontsize=12)
plt.xticks(rotation='vertical')
#plt.ylim(0,25000)
plt.show()
shop_price_dict={}
for key in np.sort(sales_train_df.shop_id.unique()).tolist():
    shop_price_dict[key]=[]
for key in shop_price_dict.keys():
    shop_price_dict[key]=sales_train_df[sales_train_df["shop_id"]==key]["item_price"].sum()
sorted_df = dict(sorted(shop_price_dict.items(), key=operator.itemgetter(1),reverse=True)[:N])
shop_price_df = pd.DataFrame(list(sorted_df.items()),columns = ['shop_id','total_price']) 
shop_price_df  

plt.figure(figsize=(12,8))

sns.barplot(x='shop_id', y="total_price", data=shop_price_df,
            order=shop_price_df.sort_values('total_price',ascending = False).shop_id,alpha=0.8,color=color[2])
plt.xlabel('Shop_id', fontsize=12)
plt.ylabel('Total_price', fontsize=12)
plt.xticks(rotation='vertical')
#plt.ylim(0,25000)
plt.show()
sales_train_df.head(2)
mon_v={'01':'Jan','02':'Feb','03':'Mar','04':'Apr','05':'May','06':'Jun','07':'Jul','08':'Aug','09':'Sep','10':'Oct','11':'Nov','12':'Dec'}
cols=['day','mon','year']
datalist = list(map(lambda x: x.split("."), sales_train_df.date)) # create list from entries in "sec" 
newdf = pd.DataFrame(data=datalist, columns=cols)   # create dataframe of new columns
sales_train_mon_df = pd.concat([sales_train_df, newdf], axis=1) 
sales_train_mon_df['mon'].replace(mon_v,inplace=True)
sales_train_mon_df.head(2)
shop_price_2013_df=sales_train_mon_df[sales_train_mon_df['year']=='2013'].groupby('shop_id')['item_price'].sum().to_frame().reset_index().sort_values(by=['item_price'], ascending=False)[:5]
shop_price_2014_df=sales_train_mon_df[sales_train_mon_df['year']=='2014'].groupby('shop_id')['item_price'].sum().to_frame().reset_index().sort_values(by=['item_price'], ascending=False)[:5]
shop_price_2015_df=sales_train_mon_df[sales_train_mon_df['year']=='2015'].groupby('shop_id')['item_price'].sum().to_frame().reset_index().sort_values(by=['item_price'], ascending=False)[:5]
fig = plt.figure(figsize=(12,5))

# Divide the figure into a 1x2 grid, and give me the first section
ax1 = fig.add_subplot(131)
plt.title('2013', fontsize=14)

ax2 = fig.add_subplot(132)
plt.title('2014', fontsize=14)

ax3 = fig.add_subplot(133)
plt.title('2015', fontsize=14)

shop_price_2013_df.plot(kind='bar', ax=ax1)
shop_price_2014_df.plot(kind='bar', ax=ax2)
shop_price_2015_df.plot(kind='bar', ax=ax3)

sales_train_mon_df.head(2)
sales_train_shop26_df=sales_train_mon_df[sales_train_mon_df['shop_id']==26].groupby('mon')['item_price'].sum().to_frame().reset_index().sort_values(by=['mon'], ascending=False)
sales_train_shop23_df=sales_train_mon_df[sales_train_mon_df['shop_id']==23].groupby('mon')['item_price'].sum().to_frame().reset_index().sort_values(by=['mon'], ascending=False)
sales_train_shop22_df=sales_train_mon_df[sales_train_mon_df['shop_id']==22].groupby('mon')['item_price'].sum().to_frame().reset_index().sort_values(by=['mon'], ascending=False)
shops_final_df=pd.merge(sales_train_shop26_df, sales_train_shop23_df, how = 'outer' ,on='mon')
shops_final_df=pd.merge(shops_final_df, sales_train_shop22_df, how = 'outer' ,on='mon')
shops_final_df.rename(columns = {'item_price_x':'shop_26', 'item_price_y':'shop_23', 'item_price' : 'shop_22'}, inplace = True) 
shops_final_df
shops_final_df.plot(kind='bar',x='mon', y=['shop_26', 'shop_23','shop_22'], figsize=(15,5), grid=True)
sales_train_26_year_df=sales_train_mon_df[sales_train_mon_df['shop_id']==26].groupby('year')['item_price'].sum().to_frame().reset_index().sort_values(by=['year'], ascending=False)
sales_train_23_year_df=sales_train_mon_df[sales_train_mon_df['shop_id']==23].groupby('year')['item_price'].sum().to_frame().reset_index().sort_values(by=['year'], ascending=False)
sales_train_22_year_df=sales_train_mon_df[sales_train_mon_df['shop_id']==22].groupby('year')['item_price'].sum().to_frame().reset_index().sort_values(by=['year'], ascending=False)

sales_train_23_year_df
plt.figure(figsize=(12,8))
# plot chart
ax1 = plt.subplot(131, aspect='equal')
sales_train_26_year_df.plot(kind='pie', y = 'item_price', ax=ax1, autopct='%1.1f%%',title='shop_26' ,
 startangle=90, shadow=False, labels=sales_train_26_year_df['year'], legend = False, fontsize=14)
ax2 = plt.subplot(132, aspect='equal')
sales_train_23_year_df.plot(kind='pie', y = 'item_price', ax=ax2, autopct='%1.1f%%', title='shop_23' ,
 startangle=90, shadow=False, labels=sales_train_23_year_df['year'], legend = False, fontsize=14)
ax3 = plt.subplot(133, aspect='equal')
sales_train_22_year_df.plot(kind='pie', y = 'item_price', ax=ax3, autopct='%1.1f%%', title='shop_22' ,
 startangle=90, shadow=False, labels=sales_train_22_year_df['year'], legend = False, fontsize=14)

plt.show()