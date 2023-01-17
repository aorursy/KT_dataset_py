# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
new_y = pd.read_csv('/kaggle/input/a-year-of-pumpkin-prices/new-york_9-24-2016_9-30-2017.csv')

new_y['Date'] = pd.to_datetime(new_y['Date'])

columbia = pd.read_csv('/kaggle/input/a-year-of-pumpkin-prices/columbia_9-24-2016_9-30-2017.csv')

columbia['Date'] = pd.to_datetime(columbia['Date'])

chicago = pd.read_csv('/kaggle/input/a-year-of-pumpkin-prices/chicago_9-24-2016_9-30-2017.csv')

chicago['Date'] = pd.to_datetime(chicago['Date'])

dallas = pd.read_csv('/kaggle/input/a-year-of-pumpkin-prices/dallas_9-24-2016_9-30-2017.csv')

dallas['Date'] = pd.to_datetime(dallas['Date'])



# chicago = chicago.set_index("Date")

# dallas = dallas.set_index("Date")

# new_y = new_y.set_index("Date")

# columbia = columbia.set_index("Date")



cities = [chicago, dallas, new_y, columbia]

df = pd.concat(cities, axis = 0)



df.reset_index(inplace = True)



df.dropna(thresh = int(df.shape[0]*.9), axis = 1, inplace = True)

df.head()
city = []

counts = []

for origin_place in df['Origin'].unique():

    origin = df.copy()[df['Origin']==origin_place]

    count = origin['Origin'].count() 

    city.append(origin_place)

    counts.append(count)

# print(counts)

# print(city)



df_max_producer = pd.DataFrame()



df_max_producer['City'] = city

df_max_producer['Number of Pumpkins'] = counts

# df_max_producer

max_producer = df_max_producer[['City','Number of Pumpkins']][df_max_producer['Number of Pumpkins']==df_max_producer['Number of Pumpkins'].max()]

print('City with maximum production of pumpkins : ')

print(max_producer)

# create a dictonary

item_size_dir = {'lge' : 4, 'xlge' : 5, 'med-lge' : 3, 'med' : 2, 'sml' : 1, 'jbo' : 6, 'exjbo' : 7}

df['Size_num'] = df['Item Size'].map(item_size_dir)

print(df.tail())
df['Avg_price'] = df.apply(lambda row : (row['Low Price']+ row['High Price'] + row['Mostly High']+ row['Mostly Low'])/4, axis = 1)

df.head()         

df.corr()
# finding the corelation between size and price

cor = ['Size_num', 'Avg_price']

df_cor = df[cor]

df_cor.corr()

# which pumpkin variety is most expensive

price_variety_df = pd.DataFrame()

price_variety_df1 = pd.DataFrame()

variety = []

price_var = []

price_var1 = []



for var in df['Variety'].unique():

    variety.append(var)

    var_df = df.copy()[df['Variety']==var]

    price = var_df['High Price'].max()

    price1 = var_df['Low Price'].min()

    price_var.append(price)

    price_var1.append(price1)

#     print(price_var1)

    

price_variety_df['Variety'] = variety

price_variety_df['Price'] = price_var

print(price_var)

price_variety_df1['Variety'] = variety

price_variety_df1['Price'] = price_var1

print(price_var1)

highest_df1 = price_variety_df[['Variety', 'Price']][price_variety_df['Price'] == price_variety_df['Price'].max()]

lowest_df2 = price_variety_df1[['Variety', 'Price']][price_variety_df1['Price'] == price_variety_df1['Price'].min()]

print('Most Expensive Pumpkin Variety')

print(highest_df1)

print('Lest Expensive Pumpkin Variety')

print(lowest_df2)