# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import pandas as pd 

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns 



import warnings

# current version of seaborn generates a bunch of warnings that we'll ignore

warnings.filterwarnings('ignore')

sns.set_style('whitegrid')



import missingno as msno # missing data visualization module for Python

import pandas_profiling



import gc

import datetime

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/ecomm_data.csv", encoding = "ISO-8859-1")  #sometimes the encoding can be weird in csv files



# In pandas you can also use read_excel (and you dont need to pay for a driver to read excel..take that, SAS!), and read_sas 
df.head()

df.info()           #info() returns information about the dataframe aka proc contents

df.describe()       #describe() returns summary stats about numeric fields for a dataframe. You can pass in a specific field



df.isnull().sum().sort_values(ascending=False)    # isnull() detects nulls values. Can use isna() to detect NA values



# check out the rows with missing values

df[df.isnull().any(axis=1)].head()
# change the column names   -  use rename(), You can pass in a dictionary of columns to be renamed as keys, with their new names as values

df.rename(index=str, columns={'InvoiceNo': 'invoice_num',

                              'StockCode' : 'stock_code',

                              'Description' : 'description',

                              'Quantity' : 'quantity',

                              'InvoiceDate' : 'invoice_date',

                              'UnitPrice' : 'unit_price',

                              'CustomerID' : 'cust_id',

                              'Country' : 'country'}, inplace=True)

df.head()



# Try this yourself. Use rename() and pass in a dictionary of new column names to name them back.
# change the invoice_date format - String to Timestamp format.  to_datetime() converts a field to a datetime

df['invoice_date'] = pd.to_datetime(df['invoice_date'], format='%m/%d/%Y %H:%M')



# can also use pd.to_numeric, or astype()
# change description - UPPER case to LOWER case - note we are working with a string so we use str.

# Note in this example we are using df.description (dot notation) instead of df['description'](bracket notation).

# These are largely interchangeable



df.description = df.description.str.lower()    # this is the same as df['description'] = df['description'].str.lower()

df.head()
# Make a copy of df without missing values. dropna() removes any NA value records

df_new = df.dropna()
# change columns tyoe - String to Int type  - astype() is the function to cast data as a specific type

df_new['cust_id'] = df_new['cust_id'].astype('int64')

df_new.head()

df_new.info()
# Subset the dataframe for all records where quantity > 0. Remember we access columns in the dataframe using brackets

df_new = df_new[df_new.quantity > 0]
df_new['amount_spent'] = df_new['quantity'] * df_new['unit_price']
# insert() adds new columns to a dataframe. loc= specifies where the column goes

# Dont worry about using lambdas, or even any of this part really. Just run it and check out the new columns added to the df



df_new.insert(loc=2, column='year_month', value=df_new['invoice_date'].map(lambda x: 100*x.year + x.month))



# You can use dt.month on a datefield to extract the month. Can also use dt.hour to get the hour, dt.dayofweek to get the day of week, etc

df_new.insert(loc=3, column='month', value=df_new.invoice_date.dt.month)    



# +1 to make Monday=1.....until Sunday=7

df_new.insert(loc=4, column='day', value=(df_new.invoice_date.dt.dayofweek)+1)

df_new.insert(loc=5, column='hour', value=df_new.invoice_date.dt.hour)



df_new.head()
group_country_orders = df_new.groupby('country')['invoice_num'].count().sort_values()  # group by country, and count the invoice numbers

print(group_country_orders)



# How many unique customers did we get from each country?

group_customers = df_new.groupby('country')['cust_id'].nunique().sort_values(ascending=False)  #nunique() = number of unique

print(group_customers)
# here we are grouping by both stock code and description, doing so by passing in a list of the columns to group by.

best_products = df_new.groupby(by=['stock_code','description'], as_index=False)['amount_spent'].sum().sort_values(by='amount_spent',ascending=False)  

print(best_products)



# Ok so looks like the birdies product is our big winner. Lets subset the dataframe to examine orders where customers bought this product

birdies = df_new[df_new['description'] == 'paper craft , little birdie']

birdies.head()



#someone bought almost 81,000 of them in a single order. lol.
df_new.groupby('invoice_num')['day'].unique().value_counts().sort_index()



# Now try orders by hour

df_new.groupby('invoice_num')['hour'].unique().value_counts()
df_new.unit_price.describe()
country_spend = df_new.groupby('country')['amount_spent'].sum().sort_values(ascending=False)

print(country_spend)