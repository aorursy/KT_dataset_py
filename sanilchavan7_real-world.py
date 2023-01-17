# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input/Pandas-Data-Science-Tasks-master/SalesAnalysis'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
import pandas as pd
import glob

path = r'/kaggle/input/Pandas-Data-Science-Tasks-master/SalesAnalysis/Sales_Data/' 
all_files = glob.glob(path + "/*.csv")

li = []

for filename in all_files:
    df = pd.read_csv(filename, index_col=None, header=0)
    li.append(df)
frame = pd.concat(li, axis=0, ignore_index=True)
frame.head()
frame.to_csv('all_months.csv')
#Read the all months combine data
all_data = pd.read_csv('/kaggle/input/Pandas-Data-Science-Tasks-master/SalesAnalysis/Output/all_data.csv')
print(all_data.shape)
#Removing nan values in the dataset
all_data = all_data.dropna(how='all')
all_data.isna().count()

all_data['Months'] = pd.to_datetime(all_data['Order Date'],  errors='coerce')
#to get particularly the month
all_data['Months'] = all_data.Months.dt.month

#CREATE A CLOUMN CALLED Month
#dont forget to add errors='coerce'
all_data['Quantity Ordered'] = pd.to_numeric(all_data['Quantity Ordered'],errors='coerce')
all_data['Price Each'] = pd.to_numeric(all_data['Price Each'],errors='coerce')
all_data['Revenue'] = all_data['Quantity Ordered']*all_data['Price Each']
all_data.head()
sales = all_data.groupby('Months').sum()
print(sales['Revenue'])
#CREATE A CITY COLUMN 
all_data['City'] = all_data['Purchase Address'].str.split(',')
all_data = all_data[all_data['Order Date'].str[0:2]!='Or']
all_data['City'] = all_data['Purchase Address'].apply(lambda x : x.split(',')[1])


#If you want revenue in highest city but also can extract revenue from it
pd.set_option('display.float_format', lambda x: '%.5f' % x)



revenue = all_data.groupby('City').sum()
print(revenue)
name_cities = [city for city,df in all_data.groupby('City')]
plt.figure(figsize=(15,8))#also can use parameter as rotation=vertical in xticks

plt.bar(name_cities,revenue['Revenue'])
plt.xticks(name_cities)
plt.show()
product = [Product for Product,df in all_data.groupby('Product')]
sales_of_product = all_data.groupby('Product')['Quantity Ordered'].sum()

plt.figure(figsize=(15,8))
plt.bar(product,sales_of_product)
plt.xticks(rotation='vertical')
plt.show()


#We plot the mean of the price of product so we conclude that which product is sold due to its price
prices = all_data.groupby('Product').mean()
print(prices)
prices = all_data.groupby('Product').mean()['Price Each']
plt.figure(figsize=(15,8))

fig, ax1 = plt.subplots(figsize=(15,8))


ax2 = ax1.twinx()
ax1.bar(product,sales_of_product)
ax2.plot(product,prices, 'b-')

ax1.set_xlabel('Products Name')
ax1.set_ylabel('Units Sold', color='g')
ax2.set_ylabel('Mean Prices', color='b')
ax1.set_xticklabels(product,rotation='vertical',size=8)
plt.show()
#We create a time(hour) column in all_data
all_data['Hour'] = pd.to_datetime(all_data['Order Date'],  errors='coerce')
all_data['Hour'] = all_data.Hour.dt.hour
hours = [Hour for Hour,df in all_data.groupby('Hour')]
print(hours)
quantity_ordered = all_data.groupby(['Hour']).count()
print(quantity_ordered)
plt.xticks(hours)
plt.plot(hours,quantity_ordered)






















































































































































































































































































































