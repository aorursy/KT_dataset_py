import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline
import os
files = [file for file in os.listdir('../input/monthly-sales-2019')]



for file in files:

    print(file)
df = pd.DataFrame()



for file in files:

    filedf =pd.read_csv('../input/monthly-sales-2019/'+file)

    df = pd.concat([df,filedf])

    

df.head()
df.to_csv("all_months_data.csv",index=False)
all_data = pd.read_csv('all_months_data.csv')
all_data.head()
len(all_data)
all_data.isnull().sum()
all_data = all_data.dropna(how='any')

all_data.head()
all_data = all_data[all_data['Order Date'].str[0:2] != 'Or']

all_data.head()
all_data['Month'] = all_data['Order Date'].str[0:2]

all_data['Month'] = all_data['Month'].astype('int32')
all_data['Quantity Ordered'] = pd.to_numeric(all_data['Quantity Ordered'])

all_data['Price Each'] = pd.to_numeric(all_data['Price Each'])

all_data['Order Date'] = pd.to_datetime(all_data['Order Date'])
all_data.head()
all_data['Hour'] = all_data['Order Date'].dt.hour
def get_city(address):

    return address.split(',')[1]

def get_state(address):

    return address.split(',')[2].split(' ')[1]



all_data['City'] = all_data['Purchase Address'].apply(lambda x:get_city(x)+' ('+get_state(x)+')')

all_data.head()
all_data['Sales'] = all_data['Quantity Ordered']*all_data['Price Each']

all_data.head()
months = range(1,13)



bymonth = all_data.groupby('Month').sum()



plt.bar(months,bymonth['Sales'])

plt.xticks(months)

plt.ylabel('Sales in $')

plt.xlabel('Month number')

plt.show()
all_data['Hour'] = all_data['Order Date'].dt.hour

all_data['Minute'] = all_data['Order Date'].dt.hour
all_data.head()
hours = [hour for hour, df in all_data.groupby('Hour')]

plt.plot(hours, all_data.groupby(['Hour']).count())

plt.xticks(hours)

plt.xlabel('time (hour)')

plt.ylabel('Number of orders')

plt.grid()

plt.show()
citysales =all_data[['City','Sales']]
results = all_data.groupby('City').sum()
cities = [city for city, df in all_data.groupby('City')]



plt.bar(cities, results['Sales'],width =0.7)

plt.xticks(cities,rotation='vertical')

plt.xlabel('Cities')

plt.ylabel('Total Sales $')

plt.show()
product_group = all_data.groupby('Product')



quantity_ordered = product_group.sum()['Quantity Ordered']



products = [p for p, df in product_group]



prices = all_data.groupby('Product').mean()['Price Each']
fig, ax1 = plt.subplots()



ax2 = ax1.twinx()

ax1.bar(products, quantity_ordered)

ax2.plot(products, prices, 'b-',color='red',markersize=10, linewidth=4, linestyle='dashed')



ax1.set_xlabel('Product name')

ax1.set_ylabel('Quantity ordered',color='blue')

ax2.set_ylabel('Price ($)',color='red')

ax1.set_xticklabels(products, rotation = 'vertical',size=8)



plt.show()

df = all_data[all_data['Order ID'].duplicated(keep=False)]

df.head(5)
df['Grouped'] = df.groupby('Order ID')['Product'].transform(lambda x:','.join(x))
df['Grouped'].value_counts()