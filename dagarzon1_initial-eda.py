# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

l = []

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        l.append(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.

l
items = pd.read_csv("/kaggle/input/competitive-data-science-predict-future-sales/items.csv")

shops_data = pd.read_csv("/kaggle/input/competitive-data-science-predict-future-sales/shops.csv")

sales_train = pd.read_csv("/kaggle/input/competitive-data-science-predict-future-sales/sales_train.csv")

test = pd.read_csv("/kaggle/input/competitive-data-science-predict-future-sales/test.csv")

sample_submission = pd.read_csv("/kaggle/input/competitive-data-science-predict-future-sales/sample_submission.csv")

item_category = pd.read_csv("/kaggle/input/competitive-data-science-predict-future-sales/item_categories.csv")
cities = []

types = []



for i in shops_data.shop_name.str.split('"'):

    j = i[0].split(' ')

    cities.append(j[0])

    types.append(j[1])

cities = np.array(cities)

types = np.array(types)



non = np.array(['', '(Плехановская,', 'Магазин', 'Орджоникидзе,', 'Посад'])

types = np.where(np.isin(types,non), 'None', types)

shops_data['city'] = cities

shops_data['type_shop'] = types 
first = []

for i in item_category.item_category_name.str.split(" "):

    first.append(i[0])

item_category['new_category'] = first

items = items.merge(item_category, on='item_category_id', how='left')
plt.figure(figsize=(13,5))



plt.subplot(121)

plt.scatter(sales_train.shop_id, sales_train.item_id)

plt.xlabel('shop_id')

plt.ylabel('item_id')

plt.title('train_data')





plt.subplot(122)

plt.scatter(test.shop_id, test.item_id)

plt.xlabel('shop_id')

plt.ylabel('item_id')

plt.title('test_data')

plt.show()
plt.figure(figsize=(13,10))



plt.subplot(221)

plt.hist(sales_train.shop_id, bins=25)

plt.xlabel('shop_id')

plt.ylabel('counts')

plt.title('train_data')



plt.subplot(222)

plt.hist(test.shop_id, bins=25)

plt.ylabel('counts')

plt.xlabel('shop_id')

plt.title('test_data')



plt.subplot(223)

plt.hist(sales_train.item_id, bins=25)

plt.xlabel('item_id')

plt.ylabel('counts')

plt.title('train_data')



plt.subplot(224)

plt.hist(test.item_id, bins=25)

plt.ylabel('counts')

plt.xlabel('item_id')

plt.title('test_data')



plt.show()
test.nunique()
sales_train.nunique()
sales_train['date'] = pd.to_datetime(sales_train.date, format='%d.%m.%Y')

sales_train['year'] = sales_train.date.dt.year

sales_train['month'] = sales_train.date.dt.month
sales_train = sales_train[sales_train.shop_id.isin(test.shop_id.unique())]

sales_train = sales_train[sales_train.item_id.isin(test.item_id.unique())]
sales_train = sales_train.merge(items[['item_id', 'item_category_id', 'new_category']], on='item_id', how='left')

sales_train = sales_train.merge(shops_data[['city', 'type_shop', 'shop_id']], on='shop_id', how='left')



sales_train.item_cnt_day = sales_train.item_cnt_day.fillna(0).clip(0,20)
import warnings

warnings.filterwarnings("ignore")



shops = sales_train.shop_id.unique()

shops = np.sort(shops)

pos = 1

plt.figure(figsize=(20,85))



for j in shops:

    k = sales_train[sales_train.shop_id == j].groupby(['new_category', 'date_block_num'], as_index=False).sum()



    for i in k.new_category.unique():

        x = k.date_block_num[k.new_category == i]

        y = k.item_cnt_day[k.new_category == i]    

        plt.subplot(int(len(shops)/3), 3, pos)     

        if j == 36:

            plt.scatter(x,y, label=i)

        else:

            plt.plot(x,y, label=i)

    

    m = sales_train[sales_train.shop_id == j].groupby('date_block_num', as_index=False).sum()

    y_t = m.item_cnt_day

    x_t = m.date_block_num.unique()

    if j == 36:

        plt.scatter(x_t, y_t, label='total', color='black' )

    else:

        plt.plot(x_t, y_t, label='total', color='black' )

    plt.xlabel('time')

    plt.ylabel('quantity')

    plt.xticks(np.arange(0, 33, 3))

    plt.title("shop_id = "+ str(j))

    plt.legend()

    pos += 1

    

plt.show()
plt.figure(figsize=(20,110))

pos = 1

for i in np.sort(sales_train.item_category_id.unique()):

    plt.subplot(20,3,pos)

    x = sales_train.date_block_num[sales_train.item_category_id == i].unique()

    x = np.sort(x)

    y = sales_train[sales_train.item_category_id == i].groupby('date_block_num').sum().item_cnt_day

    y /= np.linalg.norm(y)

    plt.plot(x,y, label='cnt')

    

    y = sales_train[sales_train.item_category_id == i].groupby('date_block_num').mean().item_price

    y_std = sales_train[sales_train.item_category_id == i].groupby('date_block_num').std().item_price.values

    y_std /= np.linalg.norm(y)

    y /= np.linalg.norm(y)

    plt.plot(x,y, label='price')

    plt.errorbar(x,y, yerr=y_std)

    plt.xlabel('time')

    plt.ylabel(' normalized quantity / average price')

    plt.xticks(np.arange(0, 33, 3))

    plt.title('item_category_id = ' + str(i))

    plt.legend()

    pos += 1

plt.show()
plt.figure(figsize=(20,110))

pos = 1

for i in np.sort(sales_train.item_category_id.unique()):

    plt.subplot(20,3,pos)

    x = sales_train.date_block_num[sales_train.item_category_id == i].unique()

    x = np.sort(x)

    y = sales_train[sales_train.item_category_id == i].groupby('date_block_num').sum().item_cnt_day

    y /= np.linalg.norm(y)

    plt.plot(x,y, label='cnt')

    

    y = sales_train[sales_train.item_category_id == i].groupby('date_block_num').mean().item_price

    y = np.diff(y)

    y /= np.linalg.norm(y)

    plt.plot(x[:-1],y, label='price')

    plt.xlabel('time')

    plt.ylabel(' normalized quantity /  change on average price')

    plt.xticks(np.arange(0, 33, 3))

    plt.title('item_category_id = ' + str(i))

    plt.legend()

    pos += 1

plt.show()
sales_train['day'] = sales_train.date.dt.day

sales_train['day_week'] = sales_train.date.dt.dayofweek
#plt.figure(figsize=(20,110))

p = [1, 0, -1]

pos = 0

for i in np.sort(sales_train.year.unique()):

    x = sales_train.month[sales_train.year == i].unique()

    x = np.sort(x)

    y = sales_train[sales_train.year == i].groupby('month').sum().item_cnt_day

    #plt.barh(x,y, label='year '+ str(i), alpha=0.5)

    

    #x = np.arange(len(x))  # the label locations

    width = 0.4  # the width of the bars



        

    plt.bar(x + p[pos] * width/2, y, width/2, label='year '+ str(i))

    pos += 1

    

    plt.xlabel('month')

    plt.ylabel('quantity')

    plt.legend()

plt.show()
plt.figure(figsize=(35,20))

pos = 1

for j in np.sort(sales_train.year.unique()):

    h = sales_train[sales_train.year == j]

    plt.subplot(2,2,pos)

    for i in np.sort(sales_train.month.unique()):

        x = h.day[h.month == i].unique()

        x = np.sort(x)

        y = h[h.month == i].groupby('day').sum().item_cnt_day

        plt.bar(x,y, label='month '+ str(i), alpha=0.5)

        plt.xlabel('day')

        plt.ylabel('quantity')

        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.title('year ' + str(j))

    pos += 1

plt.show()
plt.figure(figsize=(26,22))

p = [1, 0, -1]

pos_ = 1

labels = [ 'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday' ]

for i in np.sort(sales_train.month.unique())[:10]:

    h = sales_train[sales_train.month == i]

    plt.subplot(4, 3, pos_)

    pos = 0

    for j in np.sort(sales_train.year.unique()):

        x = h.day_week[h.year == j].unique()

        x = np.sort(x)

        y = h[h.year == j].groupby('day_week').sum().item_cnt_day

        

        x = np.arange(len(labels))  # the label locations

        width = 0.4  # the width of the bars



        

        plt.bar(x + p[pos] * width/2, y, width/2, label='year '+ str(j))

        pos += 1

        

        #plt.barh(x,y, label='year '+ str(j), alpha=0.5)

    plt.xlabel('day of the week')

    plt.ylabel('quantity')

    plt.legend()#bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.xticks(x, labels)

    plt.title('month ' + str(i))

    pos_ += 1

plt.show()