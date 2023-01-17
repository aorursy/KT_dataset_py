import numpy as np 

import pandas as pd 

import os



import matplotlib.pyplot as plt

from itertools import combinations

from collections import Counter
files = [file for file in os.listdir('../input')]



totdata = pd.DataFrame()



for file in files:

    df = pd.read_csv("../input/" + file)

    totdata = pd.concat([totdata, df])

    



totdata.to_csv("totdata.csv", index=False)
totdata = pd.read_csv("totdata.csv")

totdata.head(60)
nan_df = totdata[totdata.isna().any(axis=1)]   #finds nan values across dataframe



totdata = totdata.dropna(how='all') #drops all nan values in dataframe
totdata = totdata[totdata['Order Date'].str[0:2] != 'Or'] ## filters out wrong values 'Or'in 'Order Date' to avoid previous error



totdata['month'] = totdata['Order Date'].str[0:2]  #strips first 2 characters of values in 'Order Date'



totdata['month'] = totdata['month'].astype('int32') # Converts str to int in month column
totdata['Quantity Ordered'] = pd.to_numeric(totdata['Quantity Ordered'])



totdata['Price Each'] = pd.to_numeric(totdata['Price Each'])
totdata['Sales'] = totdata['Quantity Ordered'] * totdata['Price Each']  #calculates total sales 



totdata
results = totdata.groupby(['month']).sum() #groups by month and sums sales



results
months = range(1, 13)





plt.figure(figsize=(18,9))

plt.bar(months, results['Sales'], color='lightskyblue')

plt.xticks(months, size =14)

plt.yticks(size =14)

plt.title('Sales by Month', size=24, color='r')

plt.xlabel('Month Number', size=16)

plt.ylabel('Sales in USD ($)', size=16)



plt.show()
def get_city(address):

    return address.split(',')[1]  #function to get city from address column  



def get_state(address):

    return address.split(',')[2].split(' ')[1] #function to get Sate without zip code from address column

    

    

totdata['city'] = totdata['Purchase Address'].apply(lambda x: f"{get_city(x)} ({get_state(x)})") #creates new column adding city and State from address column





totdata
results = totdata.groupby(['city']).sum() #groups by month and sums sales



results
cities = [city for city, df in totdata.groupby('city')]



plt.figure(figsize=(18,9))

plt.bar(cities, results['Sales'], color='aquamarine')

plt.xticks(rotation=45, ha='right', size=14)

plt.yticks(size=14)

plt.title('Sales by City', size=24, color='green')

plt.xlabel('City Name', size=16)

plt.ylabel('Sales in USD ($)', size=16)



plt.show()
totdata['Order Date'] = pd.to_datetime(totdata['Order Date'])



totdata['hour'] = totdata['Order Date'].dt.hour



totdata['minute'] = totdata['Order Date'].dt.minute



totdata
hourresults = totdata.groupby(['hour']).count().sort_values('Sales', ascending=False) #groups by hour and sorts by highest sales number



hourresults
hours = [hour for hour, df in totdata.groupby('hour')]



plt.figure(figsize=(18,9))

plt.plot(hours, totdata.groupby('hour').count(), color='red')

plt.xticks(hours, size =14)

plt.yticks(size =14)

plt.grid()

plt.title('Number of Orders', size=24, color='green')

plt.xlabel('Time', size=16)

plt.ylabel('Number of Sales', size=16)



plt.show()
df = totdata[totdata['Order ID'].duplicated(keep=False)]  ### new dataframe only including duplicated order IDs



df['order group'] = df.groupby('Order ID')['Product'].transform(lambda x: ','.join(x)) ###new column concatenates products with same order ID



df = df[['Order ID', 'order group']].drop_duplicates() ### now we can keep only one order ID. Dropping the duplicates



df
count = Counter()



for row in df['order group']:

    row_list = row.split(',')

    count.update(Counter(combinations(row_list, 2)))  ### 2 items most commonly sold together

    

for key, value in count.most_common(10):

    print(key, value)
totdata.groupby(['Product']).sum().sort_values('Quantity Ordered', ascending=False)

product_group = totdata.groupby(['Product']) ### groups by product



quantity = product_group.sum()['Quantity Ordered'] ### sums up quantities



products = [product for product, df in product_group]



prices = totdata.groupby('Product').mean()['Price Each']
fig, ax1 = plt.subplots(figsize=(18,9))

ax2 = ax1.twinx()



ax1.bar(products, quantity, color='plum')

ax2.plot(products, prices, 'g--', linewidth=3)



ax1.set_xticklabels(products, rotation = 45, ha='right', size=14)



ax1.set_xlabel('Product', size=18)

ax1.set_ylabel('Quantity', size=18)

ax2.set_ylabel('Price', size=18)



plt.show()