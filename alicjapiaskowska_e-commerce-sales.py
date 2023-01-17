# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 

import warnings
warnings.filterwarnings('ignore')
sns.set_style('whitegrid')

import missingno as msno
import pandas_profiling

import gc
import datetime

%matplotlib inline
color = sns.color_palette()
df = pd.read_csv('../input/ecommerce-data/data.csv', encoding = 'ISO-8859-1')
df.head()
df.info()
# check missing values for each column 
df.isnull().sum().sort_values(ascending=False)
#% of missing values for each feature
missing_percentage = df.isnull().sum() / df.shape[0] * 100
missing_percentage
# change the format of data- String to Timestamp format
df.InvoiceDate = pd.to_datetime(df.InvoiceDate, format='%m/%d/%Y %H:%M')
df.info()



# data frame without missing values
df_new = df.dropna()
df_new.info()
df_new.isnull().sum().sort_values(ascending=False)
df_new.describe().round(2)
df_new['Country'].value_counts()
len(df_new['Country'].unique().tolist())
#Remove Quantity with negative values
df_new = df_new[df_new.Quantity > 0]
df_new.describe().round(2)
#orders made by the customers
orders=df_new.groupby(by=['CustomerID','Country'], sort=['InvoiceNo'], as_index=False)['InvoiceNo'].count()
orders.sort_values('InvoiceNo', ascending=False)
orders = df_new.groupby(by=['CustomerID','Country'], as_index=False)['InvoiceNo'].count()

plt.subplots(figsize=(20,6))
plt.plot(orders.CustomerID, orders.InvoiceNo,color ='steelblue')
plt.grid(True)
plt.xlabel('Customers ID', fontsize=15, fontname="Times New Roman")
plt.ylabel('Number of Orders',fontsize=15, fontname="Times New Roman")
plt.title('Number of Orders from different Customers', fontsize=25, color ='steelblue', fontname="Times New Roman",fontweight="bold")
plt.show()
#money spent by different customers
df_new['MoneySpent'] = df_new['Quantity'] * df_new['UnitPrice']
money_spent = df_new.groupby(by=['CustomerID','Country'], as_index=False)['MoneySpent'].sum()
money_spent.sort_values(by='MoneySpent', ascending=False)

df_new.info()
money_spent = df_new.groupby(by=['CustomerID','Country'], as_index=False)['MoneySpent'].sum()

plt.subplots(figsize=(20,6))
plt.plot(money_spent.CustomerID, money_spent.MoneySpent,color ='steelblue')
plt.grid(True)
plt.xlabel('Customers ID', fontsize=15, fontname="Times New Roman")
plt.ylabel('Money spent ($)',fontsize=15, fontname="Times New Roman")
plt.title('Money Spent from different Customers', fontsize=25, color ='steelblue', fontname="Times New Roman",fontweight="bold")
plt.show()
#money spent by customers from each country
money_spent_country = df_new.groupby(by=['Country'], as_index=False)['MoneySpent'].sum()
money_spent_country.sort_values(by='MoneySpent', ascending=False)

money_spent_country = money_spent.groupby(by=['Country'], as_index=False)['MoneySpent'].sum().sort_values(by='MoneySpent', ascending=False)
plt.subplots(figsize=(20,6))
sns.barplot(money_spent_country.Country, money_spent_country.MoneySpent,palette="Blues_r")
plt.grid(True)
plt.xlabel('Country', fontsize=15, fontname="Times New Roman")
plt.ylabel('Money spent ($)',fontsize=15, fontname="Times New Roman")
plt.title('Money Spent by customers from each country', fontsize=25, color ='steelblue', fontname="Times New Roman",fontweight="bold")
plt.xticks(rotation=80)
plt.yscale("log")
plt.show()
#amount of transactions by each country
transaction_country = df_new.groupby(by=['Country'], as_index=False)['InvoiceNo'].count()
transaction_country.sort_values(by='InvoiceNo', ascending=False)

transaction_country = df_new.groupby(by=['Country'], as_index=False)['InvoiceNo'].count().sort_values(by='InvoiceNo', ascending=False)
plt.subplots(figsize=(20,6))
sns.barplot(transaction_country.Country, transaction_country.InvoiceNo,palette="Blues_r")
plt.grid(True)
plt.xlabel('Country', fontsize=15, fontname="Times New Roman")
plt.ylabel('Transactions',fontsize=15, fontname="Times New Roman")
plt.title('Amount of transactions by each country', fontsize=25, color ='steelblue', fontname="Times New Roman",fontweight="bold")
plt.xticks(rotation=80)
plt.yscale("log")
plt.show()
df_new.info()
df_new.head()
df_new.InvoiceDate.describe()
df_new["Year"] = df_new.InvoiceDate.dt.year
df_new["Quarter"] = df_new.InvoiceDate.dt.quarter
df_new["Month"] = df_new.InvoiceDate.dt.month
df_new["Week"] = df_new.InvoiceDate.dt.week
df_new["Weekday"] = df_new.InvoiceDate.dt.weekday
df_new["Day"] = df_new.InvoiceDate.dt.day
df_new["Dayofyear"] = df_new.InvoiceDate.dt.dayofyear
df_new["Date"] = pd.to_datetime(df_new[['Year', 'Month', 'Day']])
df_new.insert(loc = 10 , column='Year_month', value=df_new['InvoiceDate'].map(lambda x: 100*x.year + x.month))

df_new.head()
monthly_sales = df_new.groupby(by=['Year_month'], as_index=False)['MoneySpent'].sum()
print(monthly_sales)
positions = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12)
yearmonth = ["Dec-2010", "Jan-2011", "Feb-2011", "Mar-2011", "Apr-2011", "May-2011",
             "Jun-2011", "Jul-1011", "Aug-2011", "Sep-2011", "Oct-2011", "Nov-2011", 
             "Dec-2011"]

monthly_sales = df_new.groupby(by=['Year_month'], as_index=False)['MoneySpent'].sum()
plt.subplots(figsize=(20,6))
sns.barplot(monthly_sales.Year_month, monthly_sales.MoneySpent,color="steelblue")
plt.grid(True)
plt.xlabel('Date', fontsize=15, fontname="Times New Roman")
plt.ylabel('Money spent ($)',fontsize=15, fontname="Times New Roman")
plt.title('Monthly money income', fontsize=25, color ='steelblue', fontname="Times New Roman",fontweight="bold")
plt.xticks(positions, yearmonth)

plt.show()