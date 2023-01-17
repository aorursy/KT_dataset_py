# Python 3 libraries
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Import data
sales = pd.read_csv('../input/sales_train.csv', parse_dates=['date'], infer_datetime_format=True, dayfirst=True)
shops = pd.read_csv('../input/shops.csv')
items = pd.read_csv('../input/items.csv')
cats = pd.read_csv('../input/item_categories.csv')
test = pd.read_csv('../input/test.csv')

# Merge 3 dataframes into consolidated df
data = pd.merge(sales, items, how='left', on='item_id')
data = pd.merge(data, cats, how='left', on='item_category_id')
data = pd.merge(data, shops, how='left', on='shop_id')

# Quick view of sales dataframe
display(sales.head(3))
display(sales.describe())

# Quick view of final dataframe
display(data.head(3))
import matplotlib.pyplot as plt
import seaborn as sns

daily_sales= data.groupby(['date']).sum().reset_index()
daily_sales=daily_sales[['date','item_cnt_day']]
daily_sales['unit']=1
display(daily_sales.head())

sns.tsplot(data=daily_sales, value='item_cnt_day', time='date', unit='unit')
plt.show()


total_item_sales= data.groupby(['item_id'])['item_cnt_day'].sum()

sns.distplot(total_item_sales)
plt.title('Distribution of Total Sales by Item')
plt.ylabel('Proportion of Dataset')
plt.xlabel('Sales Volume')
plt.show()

total_item_sales.describe()

total_shop_sales= data.groupby(['shop_id'])['item_cnt_day'].sum()

sns.distplot(total_shop_sales)
plt.title('Distribution of Total Sales by Shop')
plt.ylabel('Proportion of Dataset')
plt.xlabel('Sales Volume')
plt.show()

total_shop_sales.describe()
df = sales.groupby([sales.date.apply(lambda x: x.strftime('%Y-%m')),'item_id','shop_id']).sum()
df = df.rename(columns={'item_cnt_day': 'item_cnt_month'})
df2 = df.reset_index()
display(df2.head(5))
display(df2.dtypes)
#Baseline predictions
nov14 = df2[df2['date']=='2014-11']
nov14 = nov14[['item_id','shop_id','item_cnt_month']]
preds = pd.merge(test,nov14,on=['item_id','shop_id'], how='left')
display(preds.head())
display(preds.isnull().sum())
display(preds.describe())
#Baseline predictions
oct15 = df2[df2['date']=='2015-10']
oct15 = oct15[['item_id','shop_id','item_cnt_month']]
preds2 = pd.merge(test,oct15,on=['item_id','shop_id'], how='left')
display(preds2.head())
display(preds2.isnull().sum())
display(preds2.describe())
preds = preds[['ID','item_cnt_month']]
preds = preds.fillna(0)
preds.to_csv('submission.csv',index=False)
display(preds.head())