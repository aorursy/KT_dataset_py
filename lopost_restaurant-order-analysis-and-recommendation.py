import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the read-only "../input/" directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
prod_price_df = pd.read_csv('/kaggle/input/19560-indian-takeaway-orders/restaurant-1-products-price.csv')

prod_price_df.head(10)

orders_df = pd.read_csv('/kaggle/input/19560-indian-takeaway-orders/restaurant-1-orders.csv')

orders_df.head(10)
print(orders_df.shape)

print(orders_df.dtypes)
# convert to proper format:

try:

    orders_df['Order Date'] = pd.to_datetime(orders_df['Order Date'])

except:

    pass
# check the range of the datetime

print(orders_df['Order Date'].min())

print(orders_df['Order Date'].max())
# add additional column

orders_df['total_price'] = orders_df['Quantity'] * orders_df['Product Price']

orders_df.head(5)
orders_df[['Order Date', 'total_price']].set_index('Order Date').plot(kind='line', figsize=(15,9))

plt.title('Timeline for total order prices')

plt.ylabel('Total order $')

plt.grid()

plt.show()
orders_grouped = pd.DataFrame(orders_df.groupby(by='Order Number').total_price.sum()).sort_values('total_price')

# filter out some outliers

orders_grouped_filtered = orders_grouped[orders_grouped.total_price < 100]
orders_grouped_filtered.hist(bins = 100, rwidth = 0.8, figsize= (15,8), cumulative = True, density = True, color = '#838ac4')

plt.xticks(np.arange(0,100, 50))

plt.title('Cumulative historgram for total order price ')

plt.ylabel('% of orders')

plt.xlabel('Order $ value')

plt.show()
# add weekday column

orders_df['weekday'] = orders_df['Order Date'].dt.weekday

orders_df['weekday'] += 1



grouped_df_weekday = orders_df.groupby(by='weekday')['Order Number'].count()

grouped_df_weekday
grouped_df_weekday.plot(kind = 'bar', color = 'y', figsize= (15,6))

plt.grid()

plt.ylabel('Number or orders')

plt.title('Order volume by weekday')
# rename some columns

df_order_items = orders_df[['Order Number','Item Name']].rename(columns={'Order Number':'order_id','Item Name':'item'})

df_order_items['flag'] = 1

df_order_items.head(10)
# create a pivot table on order_id

df_order_pivot = pd.pivot_table(data = df_order_items,

              index = 'order_id',

              columns = 'item').fillna(0)

print(df_order_pivot.shape)

df_order_pivot.head(5)
# plot correlation between the dishes

df_corr_dishes = df_order_pivot.corr()

df_corr_dishes.columns = df_corr_dishes.columns.droplevel()

df_corr_dishes.reset_index(level=0, drop=True, inplace = True)

df_corr_dishes
# pick a random example as parameter for the recommendation

dish_to_recommend_to = ['Vegetable Samosa']



def return_recommended_dishes(df_corr_dishes, dish_to_recommend_to):

    '''

    Takes an input correlation dataframe and a set of at least one dish to return a sorted dataframe with recommended dishes and their score

    '''

    df_recommend_output = df_corr_dishes[dish_to_recommend_to]

    df_recommend_output['total_corr'] = df_recommend_output.mean(axis = 1)



    # sort by score

    df_recommend_output.sort_values(by= 'total_corr', ascending = False, inplace = True)



    # remove the first x items since they are the items in the list

    df_recommend_output = df_recommend_output.iloc[2:,:]



    #only show the score (total correlation)

    df_recommend_output[['total_corr']].head(10)

    

    return df_recommend_output[['total_corr']]



df_recommended_dishes = return_recommended_dishes(df_corr_dishes, dish_to_recommend_to = ['Vegetable Samosa'])

df_recommended_dishes.head(10)
df_item_popularity = pd.DataFrame(df_order_items.item.value_counts())



# how many total items were ordered

total_item_count = df_item_popularity.item.sum()



df_item_popularity['%'] = df_item_popularity.item / total_item_count

df_item_popularity.head(20)
df_order_items_list = df_order_items.groupby(by='order_id').agg({ 'item': lambda x: "','".join(x)})

df_order_items_list['item'] = "['" + df_order_items_list + "']"

df_order_items_list
from difflib import SequenceMatcher



def similar(a, b):

    return SequenceMatcher(None, a, b).ratio()
random_item = df_order_items_list.iloc[12115,:][0]

random_item
# apply the similarity function to each row, this might take a few seconds:

df_order_items_list['similarity'] = df_order_items_list['item'].apply(lambda x: similar(x, random_item))

df_order_items_list.sort_values('similarity', ascending = False, inplace= True)

df_order_items_list.iloc[3,:][0]
# create a list of items and get the frequency of each item within that list

init_list = []

for item_group in df_order_items_list.head(20).item:

    res = item_group.strip('][').split(',') 

    print(res)

    init_list = init_list + res

    

from collections import Counter



frequ_dict = Counter(init_list)

len(frequ_dict)
# create a data frame with the top recommended dishes

freq_df = pd.DataFrame.from_dict(dict(frequ_dict), orient='index').reset_index().sort_values(0, ascending = False)

freq_df_total_ratings = freq_df[0].sum()

freq_df['%_rating'] = freq_df[0] / freq_df_total_ratings

freq_df