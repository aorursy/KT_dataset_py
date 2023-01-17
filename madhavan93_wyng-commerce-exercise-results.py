# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

# print(os.listdir("../input"))



files_data = []

for file in os.listdir("../input"):

    file_data = pd.read_csv('../input/' + file)

    files_data.append(file_data)



stores_data = pd.concat(files_data)



stores_data['Discount'] = stores_data['MRP'] - stores_data['Sales Price']

stores_data['Discount Rate'] = np.where( stores_data['MRP'] != 0, 

                    (stores_data['Discount'] * 100) / stores_data['MRP'], 0)

    

stores_data.loc[(stores_data['MRP'] == 0) & (stores_data['Sales Price'] != 0), 'MRP Anamoly'] = True

stores_data.loc[~((stores_data['MRP'] == 0) & (stores_data['Sales Price'] != 0)), 'MRP Anamoly'] = False



stores_data['Sales Anamoly'] =  stores_data['Sales Price'] > stores_data['MRP']



stores_data['Sale Date'] = stores_data['Sale Date'].astype('datetime64[ns]')

stores_data['Month'] = pd.DatetimeIndex(stores_data['Sale Date']).month

stores_data['Year'] = pd.DatetimeIndex(stores_data['Sale Date']).year





def wavg(group, avg_name, weight_name):

    d = group[avg_name]

    w = group[weight_name]

    try:

        return (d * w).sum() / w.sum()

    except ZeroDivisionError:

        return 0



aggregator = {'Discount': ['count', 'sum', 'min', 'max'], 'Discount Rate': ['min', 'max', 'mean'], 'Sales Qty': ['sum', 'min', 'max'], 'Sales Price': ['sum', 'min', 'max']}

sort_by = ('Sales Qty', 'sum')

discounts_count_store_wise = stores_data[stores_data['Discount Rate'] > 0].groupby(

                'Store Code').agg(aggregator)

discounts_count_store_wise['Weighted Average Discount'] = stores_data[

    stores_data['Discount Rate'] > 0].groupby('Store Code').apply(wavg, 'Discount', 'Sales Qty')

discounts_count_store_wise['Weighted Average Sales Price'] = stores_data[

    stores_data['Discount Rate'] > 0].groupby('Store Code').apply(wavg, 'Sales Price', 'Sales Qty')

discounts_count_store_wise.sort_values(sort_by, ascending=False).to_csv('Discounts - Store Wise.csv')
grouped_discount_month_wise = stores_data[stores_data['Discount Rate'] > 0].groupby(['Year', 'Month', 'Store Code'])

grouped_discount_store_wise = stores_data[stores_data['Discount Rate'] > 0].groupby(['Store Code', 'Year', 'Month'])



discounts_month_wise = grouped_discount_month_wise.agg(aggregator)

discounts_month_wise['Weighted Average Discount'] = grouped_discount_month_wise.apply(wavg, 'Discount', 'Sales Qty')

discounts_month_wise.to_csv('Discounts - Month Store Wise.csv')



discounts_store_wise = grouped_discount_store_wise.agg(aggregator)

discounts_store_wise['Weighted Average Discount'] = grouped_discount_store_wise.apply(wavg, 'Discount', 'Sales Qty')

discounts_store_wise.to_csv('Discounts - Store Month Wise.csv')
sort_by_discount_rate = [('Discount Rate', 'mean')]

category_grouped_data = stores_data[stores_data['Discount Rate'] > 0].groupby(['Store Code', 'Category'])

discounts_category_wise = category_grouped_data.agg(aggregator)

discounts_category_wise['Weighted Average Discount'] = category_grouped_data.apply(wavg, 'Discount', 'Sales Qty')

discounts_category_wise.to_csv('Discounts Category Wise.csv')
brand_grouped = stores_data[(stores_data['Discount Rate'] > 0)].groupby(['Brand Code'])

discounts_brand_cat4_overall = brand_grouped.agg(aggregator)

discounts_brand_cat4_overall['Weighted Average Discount'] = brand_grouped.apply(wavg, 'Discount', 'Sales Qty')

discounts_brand_cat4_overall.sort_values('Weighted Average Discount', ascending=False).to_csv('Discounts Brand Wise.csv')
grouped_sales_month_wise = stores_data.groupby(['Year', 'Month', 'Store Code'])

grouped_sales_store_wise = stores_data.groupby(['Store Code', 'Year', 'Month'])

grouped_sales_store_cateogry_wise = stores_data.groupby(['Store Code', 'Category'])

grouped_sales_store_brand_wise = stores_data.groupby(['Store Code', 'Brand Code'])

aggregator2 = {'Sales Qty': ['sum', 'min', 'max'], 'Sales Price': ['sum', 'min', 'max'], 'Discount Rate': ['min', 'max', 'mean']}

sort_by = 'Weighted Sales Price Average'

sort_by_second = ('Sales Qty', 'sum')



sales_store_wise = stores_data.groupby('Store Code').agg(aggregator2)

sales_store_wise['Weighted Sales Price Average'] = stores_data.groupby('Store Code').apply(wavg, 'Sales Price', 'Sales Qty')

sales_store_wise.sort_values(sort_by, ascending=False).to_csv('Sales - Store Wise.csv')
sales_month_wise = grouped_sales_month_wise.agg(aggregator2)

sales_month_wise['Weighted Sales Price Average'] = grouped_sales_month_wise.apply(wavg, 'Sales Price', 'Sales Qty')

sales_month_wise.to_csv('Sales Month Store Wise.csv')



sales_store_month_wise = grouped_sales_store_wise.agg(aggregator2)

sales_store_month_wise['Weighted Sales Price Average'] = grouped_sales_store_wise.apply(wavg, 'Sales Price', 'Sales Qty')

sales_store_month_wise.to_csv('Sales Store Month Wise.csv')
sales_store_category_wise = grouped_sales_store_cateogry_wise.agg(aggregator2)

sales_store_category_wise['Weighted Sales Price Average'] = grouped_sales_store_cateogry_wise.apply(wavg, 'Sales Price', 'Sales Qty')

sales_store_category_wise.sort_values(sort_by, ascending=False).to_csv('Sales Category Wise.csv')
sales_store_brand_wise = grouped_sales_store_brand_wise.agg(aggregator2)

sales_store_brand_wise['Weighted Sales Price Average'] = grouped_sales_store_brand_wise.apply(wavg, 'Sales Price', 'Sales Qty')

sales_store_brand_wise.sort_values(sort_by_second, ascending=False).to_csv('Sales Store Brand Wise.csv')
margin = 0.5

stores_data['Margin'] = stores_data['Sales Price'] * margin

stores_data['Margin Quantity'] = stores_data['Margin'] * stores_data['Sales Qty']



brand_grouped = stores_data.groupby('Brand Code')

aggregator3 = {'Sales Price': ['min', 'max', 'sum', 'mean'], 'Discount Rate':['min', 'max', 'mean'], 'Sales Qty': ['min', 'max', 'sum'], 'Margin Quantity': ['min', 'max', 'sum']}

sort_by2 = ('Margin Quantity', 'sum')

sort_by3 = 'Weighted Margin Average'

brand_margin = brand_grouped.agg(aggregator3)

brand_margin['Weighted Margin Average'] = brand_grouped.apply(wavg, 'Margin', 'Sales Qty')

brand_margin.sort_values(sort_by2, ascending=False).to_csv('Brand Margin.csv')