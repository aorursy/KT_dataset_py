# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

from itertools import combinations

from collections import Counter



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# Create a list of data files from data dir.also import os

data_files_list = [file for file in os.listdir('../input/sales-data-for-eda/Sales_Data/')]



# Create an empty DF

combined_data = pd.DataFrame()



# read each file in df and then concatenate with empty DF



for file in data_files_list:

    temp_df = pd.read_csv('../input/sales-data-for-eda/Sales_Data/'+file)

    combined_data = pd.concat([combined_data, temp_df])



combined_data.to_csv('all_data_combined.csv', index=False)



#data combined. Now read the combined file directly and use for analysis

all_data = pd.read_csv('all_data_combined.csv')

# find rows with NaN across df

# nan_df = all_data[all_data.isna().any(axis=1)]

# print(nan_df.head())

# Drop nan from nan_df

all_data = all_data.dropna(how='all')



# #Check for error with unexpected field name starting 'Or'

all_data = all_data[all_data['Order Date'].str[0:2] != 'Or']

# print(temp_df.to_string())



# Create a new column called Month, extract month number from Order Data and convert Month to int.

all_data['Month'] = all_data['Order Date'].str[0:2]

all_data['Month'] = all_data['Month'].astype('int')



# Change column type of price and quantity to integer and float respectively

all_data['Quantity Ordered'] = pd.to_numeric(all_data['Quantity Ordered']) #To int

all_data['Price Each'] = pd.to_numeric(all_data['Price Each']) #To float

# print(all_data.head().to_string())

#Create column to calculate sales - quantity * price

all_data['Sales'] = all_data['Quantity Ordered'] * all_data['Price Each']

# print(all_data.head().to_string())



sales_sum = all_data.groupby('Month').sum()

print(sales_sum)



# Create bar chart using matplotlib and format

import matplotlib.pyplot as plt

months = range(1,13)

plt.bar(months, sales_sum['Sales'])

plt.title('Monthly Sales comparision chart.')

plt.xticks(months)

plt.xlabel('Month number')

plt.ylabel('Sales in USD ($)')

plt.show()

print('A: December is the best month for sales, followed by October. The reason could be festive sales where the ecommerce industry witnesses a major sales boost.')
# create a city column by extracting city name off the address



# City Splitter func definition

def get_city(address):

    return address.split(',')[1]



# State Splitter to avoid issues with multiple cities with same names

def get_state(address):

    return address.split(',')[2].split(' ')[1]



# Two option to split. Lambda function or lambda function call

all_data['City'] = all_data['Purchase Address'].apply(lambda x: x.split(',')[1])

all_data['City'] = all_data['Purchase Address'].apply(lambda x: get_city(x) + ' (' + get_state(x) + ')')

all_data['City'] = all_data['Purchase Address'].apply(lambda x: f"{get_city(x)} ({get_state(x)}) ")  #F string variation



#City sales group by

citywise_sales = all_data.groupby('City').sum()

print(citywise_sales.to_string())



# Create City wise sales bar chart

import matplotlib.pyplot as plt

cities = [city for city, df in all_data.groupby('City')]  #List comprehension to match order of data

plt.bar(cities, citywise_sales['Sales'])

plt.xticks(cities, rotation='vertical', size=8)

plt.xlabel('City Name (State)')

plt.ylabel('Sales in USD ($)')

plt.title('City-wise Sales in USA')

plt.show()



print('A: San Francisco has the highest sales in US, followed by Los Angeles.')
#Covert data to date/time object using data-time functions by making hour and minute columns

all_data['Order Date'] = pd.to_datetime(all_data['Order Date']) #Create date/time object

all_data['Hour'] = all_data['Order Date'].dt.hour

all_data['Minute'] = all_data['Order Date'].dt.minute



hours = [hour for hour, df in all_data.groupby('Hour')]  #List comprehension to match order of data #Create list of 24 hours for analysis

plt.plot(hours, all_data.groupby(['Hour']).count()) #count the orders each hours of the day

plt.xticks(hours)

plt.xlabel('Hour of the day')

plt.ylabel('Number of orders')

plt.grid()

plt.title('Hour wise order analysis')

plt.show()



print('A: 12pm and 7pm is probably the best time to advertise to maximise product purchase.')

# print(all_data.head().to_string())

#Create a new df with duplicated order. Duplicated order indicates order with two or more products. keep=False to use all occurances

combo_df = all_data[all_data['Order Date'].duplicated(keep=False)]

#Create a new column, group by orderid and combine products in a single cell comma separated

combo_df['Best Combo'] = combo_df.groupby('Order ID')['Product'].transform(lambda x: ','.join(x))

#Remove duplicated entry and show only orderid and combo column results

combo_df = combo_df[['Order ID', 'Best Combo']].drop_duplicates()

# print(combo_df.head())



#Count the occurances

from itertools import combinations

from collections import Counter



#count the combination of 2 products using the above mentioned libs.

count = Counter()

#iterate through each cell, split products in a list and count

for row in combo_df['Best Combo']:

    row_list = row.split(',')

    count.update(Counter(combinations(row_list, 2))) #group of 2. Change to 3 for group of 3s



#Most common occurance

for key, value in count.most_common(10):

    print(key, value)

    

print('\n\nA: iPhone along with Lightning Charging Cable is best selling product combination.')

product_group = all_data.groupby('Product')

quantity_ordered = product_group.sum()['Quantity Ordered']



products = [product for product, df in product_group]

plt.bar(products, quantity_ordered)

plt.xticks(products, rotation='vertical', size=8)

plt.xlabel('Product name')

plt.ylabel('Quantity ordered')

plt.title('Top selling products')

plt.show()



#Find price of the best selling products

prices = all_data.groupby('Product').mean()['Price Each']



#Find corelation between best products and their price

#Add the second axis at y for price comparision in the same graph

fig, ax1 = plt.subplots()

ax2 = ax1.twinx()

ax1.bar(products, quantity_ordered, color='g')

ax2.plot(products, prices, 'b-') #Second axis on y



ax1.set_xlabel('Product Name')

ax1.set_ylabel('Quantity Ordered', color='g')

ax1.title.set_text('Correlation betwen top selling products and its price')

ax2.set_ylabel('Price', color='b')

ax1.set_xticklabels(products, rotation='vertical', size=8)

plt.show()

print("A: The top selling product is 'AAA Batteries'. The top selling products seem to have a correlation with the price of the product. The cheaper the product higher the quantity ordered and vice versa.")