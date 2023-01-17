# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

# print(os.listdir("../input"))



weekly_data = []

for file in os.listdir("../input"):

    weekly_data = pd.read_csv('../input/' + file)



weekly_data['Date'] = weekly_data['Date'].astype('datetime64[ns]')

weekly_data['Week'] = pd.DatetimeIndex(weekly_data['Date']).week



weekly_data
for col in weekly_data.columns:

    print(col)

    print(pd.Series.nunique(weekly_data[col]))
weekly_data[weekly_data['Sales_Qty'] < 1]
weekly_grouped_data = weekly_data.groupby('Week')

aggregator = {'Sales_Qty': 'sum', 'Store_Code': pd.Series.nunique, 'State': pd.Series.nunique, 'Category': pd.Series.nunique, 'Sub_Category': pd.Series.nunique}

weekly_sales = weekly_grouped_data.agg(aggregator)

weekly_sales['Weekly Ratio'] = weekly_sales['Sales_Qty']/weekly_sales['Sales_Qty'].sum()

# weekly_sales['Weighted Ratio'] = weekly_grouped_data.apply(wavg, 'Sales_Qty', 'Store_Code')

# weekly_sales['Sales_Qty']

weekly_sales
# weekly_data[(weekly_data['Week'] == 7) | (weekly_data['Week'] == 8)].sort_values(['Week', 'Store_Code'])

# weekly_data[(weekly_data['Week'] == 7) | (weekly_data['Week'] == 8)].sort_values(['Week', 'Store_Code']).to_csv('Weekly Sorted Test.csv')



# weekly_data[((weekly_data['Week'] == 6) | (weekly_data['Week'] == 9)) & (weekly_data['Store_Code'] == 'STORE_02')].sort_values(['Week', 'Store_Code'])

weekly_data[weekly_data['Store_Code'] == 'STORE_02'].sort_values(['Week', 'Store_Code']).to_csv('Weekly Sorted Test 2.csv')
weekly_store_grouped_data = weekly_data.groupby(['Store_Code', 'Week'])

aggregator = {'Sales_Qty': 'sum', 'State': pd.Series.nunique, 'Category': pd.Series.nunique, 'Sub_Category': pd.Series.nunique}

weekly_store_sales = weekly_store_grouped_data.agg(aggregator)

weekly_store_sales.reset_index(inplace=True)

# print(weekly_store_sales)



store_wise_sums = weekly_data.groupby('Store_Code')[['Sales_Qty']].sum()

store_wise_sums.reset_index(inplace=True)

# print(store_wise_sums)

weekly_store_sales['Store Wise Ratio'] = 0.0



for store_value in weekly_data['Store_Code'].unique():

    current_store_sales = weekly_store_sales[weekly_store_sales['Store_Code'] == store_value]

    current_store_sum = store_wise_sums[store_wise_sums['Store_Code'] == store_value]['Sales_Qty'].values[0]

    # print(current_store_sales['Sales_Qty'])

    # print(current_store_sum)

    current_store_sales['Store Wise Ratio'] = current_store_sales['Sales_Qty'].div(current_store_sum)

    # print(current_store_sales)

    weekly_store_sales[weekly_store_sales['Store_Code'] == store_value] = current_store_sales



weekly_store_sales.to_csv('Weekly Store Sales.csv')    

weekly_store_sales
# Sub Category Wise Testing

weekly_store_grouped_data = weekly_data.groupby(['Store_Code', 'Sub_Category', 'Week'])

aggregator = {'Sales_Qty': 'sum'}

weekly_sub_cat_store_sales = weekly_store_grouped_data.agg(aggregator)

weekly_sub_cat_store_sales.reset_index(inplace=True)

# print(weekly_sub_cat_store_sales)



sub_cat_store_wise_sums = weekly_data.groupby(['Store_Code', 'Sub_Category'])[['Sales_Qty']].sum()

sub_cat_store_wise_sums.reset_index(inplace=True)

# print(sub_cat_store_wise_sums)

weekly_sub_cat_store_sales['SubCat Store Wise Ratio'] = 0.0



for sub_category_value in weekly_data['Sub_Category'].unique():

    for store_value in weekly_data['Store_Code'].unique():

        if len((sub_cat_store_wise_sums[(sub_cat_store_wise_sums['Sub_Category'] == sub_category_value) & 

                                    (sub_cat_store_wise_sums['Store_Code'] == store_value)]['Sales_Qty']).index) > 0:

            current_sub_cat_store_sales = weekly_sub_cat_store_sales[(weekly_sub_cat_store_sales['Sub_Category'] == 

                                    sub_category_value) & (weekly_sub_cat_store_sales['Store_Code'] == store_value)]



            current_sub_cat_store_sum = sub_cat_store_wise_sums[(sub_cat_store_wise_sums['Sub_Category'] == sub_category_value) & 

                                    (sub_cat_store_wise_sums['Store_Code'] == store_value)]['Sales_Qty'].values[0]

            # print(current_sub_cat_store_sales['Sales_Qty'])

            # print(current_sub_cat_store_sum)

            current_sub_cat_store_sales['SubCat Store Wise Ratio'] = current_sub_cat_store_sales['Sales_Qty'].div(current_sub_cat_store_sum)

            # print(current_store_sales)

            weekly_sub_cat_store_sales[(weekly_sub_cat_store_sales['Sub_Category'] == sub_category_value) & 

                               (weekly_sub_cat_store_sales['Store_Code'] == store_value)] = current_sub_cat_store_sales

        



weekly_sub_cat_store_sales.to_csv('Weekly Store SubCat Sales.csv')    

weekly_sub_cat_store_sales
store2_data = weekly_data[weekly_data['Store_Code'] == 'STORE_02']

# print(store2_data)

print(store2_data.groupby(['Category', 'Sub_Category']).agg({'Sales_Qty':'mean'}))



new_data = pd.DataFrame({

    'Date': ['2018-02-12', '2018-02-12', '2018-02-12', '2018-02-12', '2018-02-12', '2018-02-12', '2018-02-12', '2018-02-19', '2018-02-19', '2018-02-19', '2018-02-19', '2018-02-19', '2018-02-19', '2018-02-19'],

    'Store_Code': ['STORE_02', 'STORE_02', 'STORE_02', 'STORE_02', 'STORE_02', 'STORE_02', 'STORE_02', 'STORE_02', 'STORE_02', 'STORE_02', 'STORE_02', 'STORE_02', 'STORE_02', 'STORE_02'],

    'State': ['TG', 'TG', 'TG', 'TG', 'TG', 'TG', 'TG', 'TG', 'TG', 'TG', 'TG', 'TG', 'TG', 'TG'],

    'Category': ['CAT1', 'CAT1', 'CAT2', 'CAT2', 'CAT2', 'CAT2', 'CAT2', 'CAT1', 'CAT1', 'CAT2', 'CAT2', 'CAT2', 'CAT2', 'CAT2'],

    'Sub_Category': ['SUBCAT1', 'SUBCAT2', 'SUBCAT4', 'SUBCAT5', 'SUBCAT6', 'SUBCAT7', 'SUBCAT8', 'SUBCAT1', 'SUBCAT2', 'SUBCAT4', 'SUBCAT5', 'SUBCAT6', 'SUBCAT7', 'SUBCAT8'],

    'Sales_Qty': [20, 2, 2, 3, 2, 2, 2, 20, 2, 2, 3, 2, 2, 2],

    'Week': [7, 7, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 8, 8]

}, index=range(5181, 5181+14, 1))



new_data['Date'] = new_data['Date'].astype('datetime64[ns]')



weekly_data_store_added = weekly_data.append(new_data)

weekly_data_store_added
weekly_store_grouped_data = weekly_data.groupby(['Store_Code', 'Week'])

aggregator = {'Sales_Qty': 'sum', 'State': pd.Series.nunique, 'Category': pd.Series.nunique, 'Sub_Category': pd.Series.nunique}

weekly_store_sales = weekly_store_grouped_data.agg(aggregator)

weekly_store_sales.reset_index(inplace=True)

# print(weekly_store_sales)



store_wise_sums = weekly_data.groupby('Store_Code')[['Sales_Qty']].sum()

store_wise_sums.reset_index(inplace=True)

# print(store_wise_sums)

weekly_store_sales['Store Wise Ratio'] = 0.0



for store_value in weekly_data['Store_Code'].unique():

    current_store_sales = weekly_store_sales[weekly_store_sales['Store_Code'] == store_value]

    current_store_sum = store_wise_sums[store_wise_sums['Store_Code'] == store_value]['Sales_Qty'].values[0]

    # print(current_store_sales['Sales_Qty'])

    # print(current_store_sum)

    current_store_sales['Store Wise Ratio'] = current_store_sales['Sales_Qty'].div(current_store_sum)

    # print(current_store_sales)

    weekly_store_sales[weekly_store_sales['Store_Code'] == store_value] = current_store_sales



weekly_store_sales.to_csv('Weekly Store Sales 2.csv')    

weekly_store_sales