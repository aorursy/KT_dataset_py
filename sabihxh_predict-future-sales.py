%matplotlib notebook



import numpy as np

import pandas as pd

import matplotlib

import matplotlib.pyplot as plt



import os

print(os.listdir("../input"))



matplotlib.get_backend()
os.listdir('../working/')
fp = lambda filename: os.path.join('../input/', filename)

test = pd.read_csv(fp('test.csv'))

item_categories = pd.read_csv(fp('item_categories.csv'))

sales_train = pd.read_csv(fp('sales_train.csv'))

items = pd.read_csv(fp('items.csv'))

shops = pd.read_csv(fp('shops.csv'))
# Target variable is float64, it should be int.

sales_train['item_cnt_day'] = sales_train['item_cnt_day'].astype(int)
# Get value count of number of sales

item_value_count = sales_train['item_cnt_day'].value_counts().sort_index()

item_value_count.head()
# Create new date related columns

def to_yyyymmdd(date):

    """

    Converts date from format dd.mm.yyyy to yyyymmdd

    """

    date_list = date.split('.')

    return date_list[2] + date_list[1] + date_list[0]

    

sales_train['year'] = sales_train['date'].apply(lambda date: date[6:]).astype(int)

sales_train['month'] = sales_train['date'].apply(lambda date: date[3:5]).astype(int)

sales_train['date_yyyymmdd'] = sales_train['date'].apply(lambda date: to_yyyymmdd(date))
sales_train = sales_train.sort_values(by='date_yyyymmdd')

sales_train.head()
# Aggregate data

monthly_sales = sales_train.groupby('month').agg({'item_cnt_day': np.sum})

# month_map = {1:'J', 2:'F', 3:'M', 4:'A', 5:'M', 6:'J', 7:'J', 8:'A', 9:'S', 10:'O', 11:'N', 12:'D'}
fig = plt.figure(figsize=(8, 6))



plt.bar(

    monthly_sales.index, 

    monthly_sales['item_cnt_day'], 

    tick_label=list('JFMAMJJASOND'), 

    color='#79ccb3'

)



plt.title('Products Sold by Month')

plt.xlabel('Month')

plt.ylabel('Products Sold')



# Remove top and right border

plt.gca().spines['top'].set_visible(False)

plt.gca().spines['right'].set_visible(False)



plt.show()

fig.savefig('products_sold_by_month.png')

