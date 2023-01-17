# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import re

import matplotlib.pyplot as plt

%matplotlib inline



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
flipkart_data = pd.read_csv(r"/kaggle/input/filpkart-onlineorders/OnlineOrders_of_a_ecommerce_website.csv")
flipkart_data.head()
flipkart_data.rename(columns={'crawl_timestamp': 'Timestamp',

                              'product_name': 'Product_Name',

                             'product_category_tree': 'Product_Category_Tree',

                             'retail_price': 'Retail_Price',

                             'discounted_price': 'Discounted_Price',

                             'brand': 'Brand'}, inplace=True)
flipkart_data.head()
flipkart_data['Category'] = flipkart_data['Product_Category_Tree'].apply(lambda x: re.split('\[]*|\"|\>>|\,', x)[2])
flipkart_data.head(5)
flipkart_data.drop(['Product_Category_Tree'], axis = 1, inplace= True)
flipkart_data.head()
flipkart_data['Timestamp'] = flipkart_data['Timestamp'].apply(lambda x: x.split('+')[0])
flipkart_data.head()
# Save the data as csv file

flipkart_data.to_csv('fkartDataset.csv', index=False)
flkart_data = pd.read_csv('fkartDataset.csv')

flkart_data.head()
flkart_data.isnull().sum()
#Adding the month column

flkart_data['Month'] = pd.to_numeric(pd.DatetimeIndex(flkart_data['Timestamp']).month)

flkart_data.head()
totalsum = flkart_data.groupby('Month').sum()

totalsum 
months = range(1, 7)

plt.bar(months, totalsum['Discounted_Price'])

plt.xticks(months)

plt.xlabel("Months")

plt.ylabel('Sales in INR')

plt.show()
flkart_data['Timestamp'] = pd.to_datetime(flkart_data['Timestamp'])
flkart_data['Hour'] = flkart_data['Timestamp'].dt.hour

flkart_data['Minute'] = flkart_data['Timestamp'].dt.minute
flkart_data.head()
hours = [hour for hour, df in flkart_data.groupby('Hour')]



plt.plot(hours, flkart_data.groupby(['Hour']).count())

plt.xticks(hours)

plt.xlabel('Hour')

plt.ylabel('Number of Orders')

plt.grid()

plt.show()
# So our target is to look after duplicates rows

dups_category = flkart_data.pivot_table(index=['Category'], aggfunc='size')
print(dups_category.nlargest(6))



x =list(range(1,7))



fig, ax = plt.subplots()

bar = sns.barplot(data=flkart_data, x=x , y=dups_category.nlargest(6), edgecolor="white")

ax.set_xticklabels(["Clothes", "Jewel", 'Mobile&Accessories', 'Home Decor', 'Footwear', 'Tools&Hardware'], rotation=90)

plt.show();
# So our target is to look after duplicates rows

dups_product = flkart_data.pivot_table(index=['Product_Name'], aggfunc='size')



print(dups_product.nlargest(10))

items = range(10)



x =list(range(1,11))

fig, ax = plt.subplots()

bar = sns.barplot(data=flkart_data, x=x , y=dups_category.nlargest(10), edgecolor="white")

plt.show();
