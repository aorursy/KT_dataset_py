import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import os

print(os.listdir("../input"))
shops = pd.read_csv("../input/shops.csv")  # shop_name, shop_id

item_categories = pd.read_csv("../input/item_categories.csv")  # item_category_name, item_category_id

sales_train = pd.read_csv("../input/sales_train.csv")  # date, date_block_num, shop_id, item_id, item_price, item_cnt_day

items = pd.read_csv("../input/items.csv")  # item_name, item_id, item_category_id



test = pd.read_csv("../input/test.csv")  # ID, (shop_id, item_id)

sample_submission = pd.read_csv("../input/sample_submission.csv")  # ID, item_cnt_month
print('\n# shops\n')

shops.info()

print('\n# item_categories\n')

item_categories.info()

print('\n# sales_train\n')

sales_train.info()

print('\n# items\n')

items.info()

print('\n# test\n')

test.info()
sales_train = pd.merge(sales_train, items, on=['item_id'])

sales_train = pd.merge(sales_train, item_categories, on=['item_category_id'])

sales_train = pd.merge(sales_train, shops, on=['shop_id'])

sales_train.head()
shops.drop_duplicates(keep = 'first', inplace = True)

item_categories.drop_duplicates(keep = 'first', inplace = True)

sales_train.drop_duplicates(keep = 'first', inplace = True)

items.drop_duplicates(keep = 'first', inplace = True)
pd.isnull(sales_train).sum()
sales_train.info()
sales_train_copy = sales_train.copy()
# how many items in each category in training data



fig, axs = plt.subplots(2,2,figsize=(15,12))



# Total sales variation



records_category = pd.concat([items.groupby('item_category_id')['item_id'].count(),sales_train_copy.groupby('item_category_id')['item_cnt_day'].sum()],axis=1).rename(columns={'item_id':'item_numbers','item_cnt_day':'total_sales'})  

records_category.sort_index()

records_category['average_sales'] = sales_train_copy.groupby(['item_category_id','item_id'])['item_cnt_day'].sum().groupby('item_category_id').mean().sort_index()  

records_category['sales_std'] = sales_train_copy.groupby(['item_category_id','item_id'])['item_cnt_day'].sum().groupby('item_category_id').std().sort_index() 



# Month variations



# Plot



items.groupby('item_category_id')['item_id'].count().plot(kind='bar',title='item counts in the category (training data)', ax=axs[0,0])     

sales_train_copy.groupby('item_category_id')['item_cnt_day'].sum().plot(kind='bar',title='sales of all items in the category', ax = axs[0,1]) 

records_category['average_sales'].plot(kind='bar',title='average sales of the item in the category', ax = axs[1,0]) 

records_category['sales_std'].plot(kind='bar',title='sales std of the items in the category', ax = axs[1,1]) 



print('Corelation', items.groupby('item_category_id')['item_id'].count().corr(sales_train_copy.groupby('item_category_id')['item_cnt_day'].sum())) 
records_category.sort_values(by='average_sales',ascending=False).head()
# Category sales through time



sales_train_copy_fplot = sales_train_copy.groupby(['date_block_num','item_category_id'])['item_cnt_day'].sum()

plot_df = sales_train_copy_fplot.unstack('item_category_id').loc[:]

plot_df.plot(legend=False, kind = 'line')

items[items['item_category_id']==40].head()
cate40_items_ex = [0,10,24,37,22149,22156,22160,22163]

#items in cate40_items_ex

items[items['item_id'].isin(cate40_items_ex) == True]
sales_train_copy.groupby('shop_id').sum()['item_cnt_day'].plot(kind='bar', title='total sales of each shop')
a = sales_train_copy.groupby(['shop_id','item_category_id']).sum()['item_cnt_day'].reset_index()

fig, axs = plt.subplots(1,3,figsize=(15,5))

a[a.groupby(['shop_id'])['item_cnt_day'].transform(max) == a['item_cnt_day']].reset_index()['item_category_id'].plot(kind='bar',ax=axs[0],title='Best sellinf cates in each shop')

a[a.groupby(['shop_id'])['item_cnt_day'].transform(max) == a['item_cnt_day']].reset_index().groupby('item_category_id').count()['shop_id'].plot(kind='pie', title='Ratio of Best Selling cates in 60 shops', ax=axs[1]).set_ylabel('') 

a[a.groupby(['item_category_id'])['item_cnt_day'].transform(max) == a['item_cnt_day']].reset_index().groupby('shop_id').count()['item_category_id'].plot(kind='pie', title='Ratio of Best Selling shops in 84 shops', ax=axs[2]).set_ylabel('') 
plt.figure(figsize=(10,4)) # sales of all samples

sns.boxplot(x=sales_train_copy.item_cnt_day)



plt.figure(figsize=(10,4)) # price of all samples

sns.boxplot(x=sales_train_copy.item_price)
sales_train_copy.loc[sales_train_copy['item_cnt_day'].idxmax()]
sales_train_copy.loc[sales_train_copy['item_cnt_day'].idxmin()]
sales_train_copy.loc[sales_train_copy['item_price'].idxmax()]
sales_train_copy.loc[sales_train_copy['item_price'].idxmin()]
sales_train_copy.groupby(['item_id']).sum()['item_cnt_day'].plot()
bestsale = sales_train_copy.groupby(['item_id']).get_group(sales_train_copy.groupby(['item_id']).sum()['item_cnt_day'].idxmax())

bestsale.sort_values(by='item_cnt_day', ascending=False).head()
train_keys_shop_item = sales_train_copy.groupby(['item_id','shop_id']).groups.keys()

test_keys_shop_item = test.groupby(['item_id','shop_id']).groups.keys()

print('# train_keys and test_keys size', len(list(train_keys_shop_item)), len(list(test_keys_shop_item)))

print('# intersection', len(set(list(train_keys_shop_item)) & set(list(test_keys_shop_item))))
train_keys = sales_train_copy.groupby(['item_id']).groups.keys()

test_keys = test.groupby(['item_id']).groups.keys()

print('# train_keys and test_keys size', len(list(train_keys)), len(list(test_keys)))

print('# intersection', len(set(list(train_keys)) & set(list(test_keys))))
test.isin({'item_id': list(train_keys)}).groupby('item_id').size()
test_length = len(test)

olditems_alreadylaunched = len(set(list(train_keys_shop_item)) & set(list(test_keys_shop_item)))

newitems_newlaunched = len(test.isin({'item_id': list(train_keys)}).groupby('item_id').get_group(False))

olditems_newlaunched = test_length - olditems_alreadylaunched - newitems_newlaunched



# Data to plot



labels = 'old items already launched', 'old items new launched', 'new items new launched'

sizes = [olditems_alreadylaunched, olditems_newlaunched, newitems_newlaunched]



plt.pie(sizes, labels=labels,

autopct='%1.1f%%', shadow=True, startangle=140)

plt.axis('equal')

plt.show()
olditemsshopsintest = pd.DataFrame(np.asarray(list(set(list(train_keys_shop_item)) & set(list(test_keys_shop_item)))), columns=['item_id','shop_id']) 

olditemsshopsIDintest = pd.merge(test, olditemsshopsintest)['ID']



test1 = pd.concat([test, test.isin({'item_id': list(train_keys)})['item_id'].rename('item_in_train') ], axis=1)

newitemsIDintest = test1.groupby('item_in_train').get_group(False)['ID']



olditemsnewshopIDintest = test.drop(pd.concat([olditemsshopsIDintest,newitemsIDintest]))['ID']



# Index data



olditemsnewshopintest_data = test.loc[test['ID'].isin(olditemsnewshopIDintest)]

olditemsshopsintest_data = test.loc[test['ID'].isin(olditemsshopsIDintest)]

newitemsintest_data = test.loc[test['ID'].isin(newitemsIDintest)]
# Group by items



toplot = olditemsnewshopintest_data[['shop_id','item_id']].groupby('item_id').count().rename(columns = {'shop_id':'shop_numbers_intest'}).reset_index()   



itemsoldinshops_train = sales_train_copy.groupby(['item_id','shop_id']).count().reset_index()[['item_id','shop_id']].groupby('item_id').count().rename(columns = {'shop_id':'shop_numbers_intrain'}).reset_index()[['item_id','shop_numbers_intrain']]  

toplot = pd.merge(toplot, itemsoldinshops_train)

toplot['test_train_shopsratio'] = toplot['shop_numbers_intest'] / toplot['shop_numbers_intrain']



fig,axs = plt.subplots(1,3,figsize=(15,5))

toplot.sort_values(by='shop_numbers_intrain').reset_index()['shop_numbers_intrain'].plot(ax=axs[0])

toplot.sort_values(by='shop_numbers_intest').reset_index()['shop_numbers_intest'].plot(ax=axs[1])

toplot.sort_values(by='test_train_shopsratio').reset_index()['test_train_shopsratio'].plot(ax=axs[2])

axs[0].title.set_text('shop_numbers_in_train')

axs[1].title.set_text('shop_numbers_in_test')

axs[2].title.set_text('test_train_shopsratio')
# The category ratio of this task

toplot = pd.merge(toplot, items)

toplot.groupby('item_category_id').count()['item_id'].rename(columns={'item_id':'item_count'}).plot()   
toplot[toplot['item_id'].isin([11286,13818,4240])]
fig, axs = plt.subplots(1,3,figsize=(15,5))



sales_train_copy[sales_train_copy['item_id']==4240].groupby('date_block_num').sum().rename(columns={'item_cnt_day':'monthly_sales'})['monthly_sales'].plot(ax=axs[0])      



sales_train_copy_fplot = sales_train_copy[sales_train_copy['item_id']==4240].groupby(['date_block_num','shop_id'])['item_cnt_day'].sum()

plot_df = sales_train_copy_fplot.unstack('shop_id').loc[:]

plot_df.plot(legend=False, kind = 'line', ax=axs[1])



sales_train_copy[sales_train_copy['item_id']==4240].groupby(['date_block_num','shop_id']).sum().groupby('date_block_num').mean().rename(columns={'item_cnt_day':'monthly_sales'})['monthly_sales'].plot(ax=axs[2],kind='bar')      



axs[0].set_title('Monthly sales of all shops of item 4240')

axs[1].set_title('Monthly sales of each shop of item 4240')

axs[2].set_title('Monthly average sales of the shops having sales')
fig, axs = plt.subplots(1,3,figsize=(15,5))



sales_train_copy[sales_train_copy['item_category_id']==23].groupby('date_block_num').sum().rename(columns={'item_cnt_day':'monthly_sales'})['monthly_sales'].plot(ax=axs[0])      



sales_train_copy_fplot = sales_train_copy[sales_train_copy['item_category_id']==23].groupby(['date_block_num','shop_id'])['item_cnt_day'].sum()

plot_df = sales_train_copy_fplot.unstack('shop_id').loc[:]

plot_df.plot(legend=False, kind = 'line', ax=axs[1])



sales_train_copy[sales_train_copy['item_category_id']==23].groupby(['date_block_num','shop_id']).sum().groupby('date_block_num').mean().rename(columns={'item_cnt_day':'monthly_sales'})['monthly_sales'].plot(ax=axs[2],kind='bar')      



axs[0].set_title('Monthly sales of all shops of cate 23')

axs[1].set_title('Monthly sales of each shop of cate 23')

axs[2].set_title('Monthly average cate sales of the shops having sales')
fig, axs = plt.subplots(1,3,figsize=(15,5))



sales_train_copy[sales_train_copy['item_id']==13818].groupby('date_block_num').sum().rename(columns={'item_cnt_day':'monthly_sales'})['monthly_sales'].plot(ax=axs[0],kind='bar')      



sales_train_copy_fplot = sales_train_copy[sales_train_copy['item_id']==13818].groupby(['date_block_num','shop_id'])['item_cnt_day'].sum()

plot_df = sales_train_copy_fplot.unstack('shop_id').loc[:]

plot_df.plot(legend=False, kind = 'line', ax=axs[1])



sales_train_copy[sales_train_copy['item_id']==13818].groupby(['date_block_num','shop_id']).sum().groupby('date_block_num').mean().rename(columns={'item_cnt_day':'monthly_sales'})['monthly_sales'].plot(ax=axs[2],kind='bar')      



axs[0].set_title('Monthly sales of all shops of item 13818')

axs[1].set_title('Monthly sales of each shop of item 13818')

axs[2].set_title('Monthly average sales of the shops having sales')
fig, axs = plt.subplots(1,3,figsize=(15,5))



sales_train_copy[sales_train_copy['item_category_id']==37].groupby('date_block_num').sum().rename(columns={'item_cnt_day':'monthly_sales'})['monthly_sales'].plot(ax=axs[0])      



sales_train_copy_fplot = sales_train_copy[sales_train_copy['item_category_id']==37].groupby(['date_block_num','shop_id'])['item_cnt_day'].sum()

plot_df = sales_train_copy_fplot.unstack('shop_id').loc[:]

plot_df.plot(legend=False, kind = 'line', ax=axs[1])



sales_train_copy[sales_train_copy['item_category_id']==37].groupby(['date_block_num','shop_id']).sum().groupby('date_block_num').mean().rename(columns={'item_cnt_day':'monthly_sales'})['monthly_sales'].plot(ax=axs[2],kind='bar')      



axs[0].set_title('Monthly sales of all shops of cate 37')

axs[1].set_title('Monthly sales of each shop of cate 37')

axs[2].set_title('Monthly average cate sales of the shops having sales')
sales_train_copy[sales_train_copy['item_id']==11286].groupby('date_block_num').sum().rename(columns={'item_cnt_day':'monthly_sales'})['monthly_sales'].plot(title='Monthly sales of all shops of item 11286 (only one shop)')      
fig, axs = plt.subplots(1,3,figsize=(15,5))



sales_train_copy[sales_train_copy['item_category_id']==31].groupby('date_block_num').sum().rename(columns={'item_cnt_day':'monthly_sales'})['monthly_sales'].plot(kind='bar', ax=axs[0])      



sales_train_copy_fplot = sales_train_copy[sales_train_copy['item_category_id']==31].groupby(['date_block_num','shop_id'])['item_cnt_day'].sum()

plot_df = sales_train_copy_fplot.unstack('shop_id').loc[:]

plot_df.plot(legend=False, kind = 'line', ax=axs[1])



sales_train_copy[sales_train_copy['item_category_id']==31].groupby(['date_block_num','shop_id']).sum().groupby('date_block_num').mean().rename(columns={'item_cnt_day':'monthly_sales'})['monthly_sales'].plot(ax=axs[2],kind='bar')      



axs[0].set_title('Monthly sales of all shops of cate 31')

axs[1].set_title('Monthly sales of each shop of cate 31')

axs[2].set_title('Monthly average cate sales of the shops having sales')
newitemsintest_data = pd.merge(newitemsintest_data,items[['item_id','item_category_id']],on='item_id')



fig, axs = plt.subplots(1,2,figsize=(15,8))

newitemsintest_data.groupby('item_category_id').count()['item_id'].plot(kind='bar', ax=axs[0])

newitemsintest_data.groupby('item_id').count()['shop_id'].plot(kind='bar',ax=axs[1]) # all the same



axs[0].set_title('test input counted in item_category')

axs[1].set_title('test input counted in items [based on shop]')
fig, axs = plt.subplots(1,3,figsize=(15,5))



sales_train_copy[sales_train_copy['item_category_id']==72].groupby('date_block_num').sum().rename(columns={'item_cnt_day':'monthly_sales'})['monthly_sales'].plot(ax=axs[0])      



sales_train_copy_fplot = sales_train_copy[sales_train_copy['item_category_id']==72].groupby(['date_block_num','shop_id'])['item_cnt_day'].sum()

plot_df = sales_train_copy_fplot.unstack('shop_id').loc[:]

plot_df.plot(legend=False, kind = 'line', ax=axs[1])



sales_train_copy[sales_train_copy['item_category_id']==72].groupby(['date_block_num','shop_id']).sum().groupby('date_block_num').mean().rename(columns={'item_cnt_day':'monthly_sales'})['monthly_sales'].plot(ax=axs[2],kind='bar')      



axs[0].set_title('Monthly sales of all shops of cate 72')

axs[1].set_title('Monthly sales of each shop of cate 72')

axs[2].set_title('Monthly average cate sales of the shops having sales')
sales_train_copy.groupby('item_price').size()
sales_train_copy.groupby('item_price').get_group(-1)
sales_train_copy[(sales_train_copy['item_id']==2973) & (sales_train_copy['shop_id']==32)].head()
sale2973 = sales_train_copy[sales_train_copy['item_id']==2973]

sale2973.head()
fig, axs = plt.subplots(1,2,figsize=(15,5))



sale2973.groupby('date_block_num')['item_cnt_day'].sum().plot(kind=' bar', ax=axs[0], title='monthly total sales of the item in all shops') # monthly sales of the item 



sale2973_fplot = sale2973.groupby(['date_block_num','shop_id'])['item_cnt_day'].sum()

plot_df = sale2973_fplot.unstack('shop_id').loc[:]

plot_df.plot(legend=False, kind = 'line', ax=axs[1], title='monthly total sales of the item in each shops')

fig, axs = plt.subplots(1,2,figsize=(15,5))



sale2973.groupby(['date_block_num'])['item_price'].mean().plot(kind = 'bar', ax=axs[0], title='monthly average price of all shops of the item')

pd.Series(sale2973.groupby(['date_block_num', 'shop_id'])['item_price'].mean().groupby('date_block_num').std()).plot(kind = 'bar', ax=axs[1], title='monthly price std between all the shops')     
print(sales_train_copy[sales_train_copy['item_cnt_day']<0].groupby(['item_id']).size())

print(sales_train_copy[sales_train_copy['item_cnt_day']<0].groupby(['item_id','date_block_num', 'shop_id']).size())
# Category with sales return 



fig, axs = plt.subplots(1,3,figsize=(15,5))



# item-based negative-sold counts in each category

sales_train_copy[sales_train_copy['item_cnt_day']<0].groupby(['item_category_id'])['item_id'].agg(['count']).plot(kind='bar',ax = axs[0])                               

# item negative-sold total sales in each category

sales_train_copy[sales_train_copy['item_cnt_day']<0].groupby(['item_category_id'])['item_cnt_day'].agg(['sum']).abs().plot(kind='bar', ax = axs[1])                           

# item-based

negative_totalsales = sales_train_copy[sales_train_copy['item_cnt_day']<0].groupby('item_id')['item_cnt_day'].agg('sum').abs()

negative_totalsales.plot(kind='line', ax=axs[2], title='Counts of returns of each item')



axs[0].title.set_text('Counts of items with return record in the category')

axs[1].title.set_text('Ruturn sales of items in the category')
sales_2331 = sales_train_copy[sales_train_copy['item_id']==2331]

print('positive sales', sales_2331[sales_2331['item_cnt_day']>0]['item_cnt_day'].sum())

print('negative sales', sales_2331[sales_2331['item_cnt_day']<0]['item_cnt_day'].sum())
sales_cate20 = sales_train_copy[sales_train_copy['item_category_id']==20]



# 157 items in cate 20 in training data 

fig, axs = plt.subplots(1,2,figsize=(15,5))

fig.suptitle('Total sales of the 157 items in category 20 (training data)')

sales_cate20.groupby('item_id').agg('sum')['item_cnt_day'].sort_values().plot(kind='bar', ax=axs[0])

sales_cate20.groupby('item_id').agg('sum')['item_cnt_day'].sort_values().plot(kind='pie', ax=axs[1])

# negative sales of the category - 100/157 items with negative sales in this category

fig, axs = plt.subplots(1,2,figsize=(15,5))

fig.suptitle('Total returns of the 100 items in category 20 (training data)')

sales_cate20[sales_cate20['item_cnt_day']<0].groupby('item_id').agg('sum')['item_cnt_day'].abs().sort_values().plot(kind='bar', ax=axs[0])

sales_cate20[sales_cate20['item_cnt_day']<0].groupby('item_id').agg('sum')['item_cnt_day'].abs().sort_values().plot(kind='pie', ax=axs[1])
## Among these negative-sale items in cate20 - 100 items in 157/175 items in category 20



# negative sales

sales20_negtotal = sales_cate20[sales_cate20['item_cnt_day']<0].groupby('item_id').agg('sum')['item_cnt_day'].reset_index()

# positive sales

sales20_postotal = sales_cate20[sales_cate20['item_cnt_day']>0].groupby('item_id').agg('sum')['item_cnt_day'].reset_index()

# total sales

sales20_total = sales_cate20.groupby('item_id').agg('sum')['item_cnt_day'].reset_index()



sales20_compare = pd.merge(sales20_postotal, sales20_negtotal.abs(), on='item_id')

sales20_compare.rename(columns={"item_cnt_day_x": "sales_sold", "item_cnt_day_y": "sales_return"}, inplace=True)

sales20_compare = pd.merge(sales20_compare, sales20_total, on='item_id')

sales20_compare.rename(columns={"item_cnt_day": "sales_total"}, inplace=True)

sales20_compare.fillna(0)

sales20_compare['return_ratio'] = sales20_compare['sales_return'] / sales20_compare['sales_sold']



fig, axs = plt.subplots(2,2,figsize=(15,10))

fig.suptitle('Sale records of the 100 items with returns in category 20 (training data)')

sales20_compare[['item_id','sales_sold']].plot(x='item_id',y='sales_sold', kind='bar', ax=axs[0,0])

sales20_compare[['item_id','sales_return']].plot(x='item_id',y='sales_return', kind='bar', ax=axs[0,1])

sales20_compare[['item_id','sales_total']].plot(x='item_id',y='sales_total', kind='bar', ax=axs[1,0])

sales20_compare[['item_id','return_ratio']].plot(x='item_id',y='return_ratio', kind='bar', ax=axs[1,1])