# Import needed libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
import re

pd.options.display.max_rows = None
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
# Read data in the excel file

df = pd.read_excel('../input/online-retail-data-set-from-uci-ml-repo/Online Retail.xlsx')
df.head()
df.shape
df.info()
df.describe()
# Check null values
df.isnull().sum()
# Check number of unique values
df.nunique()
# Check each stock code has only one description
df.groupby('StockCode').apply(lambda x: x['Description'].unique()).head()
# Number of invoices for each country
df.groupby(['Country']).count() ['InvoiceNo']
# Delete rows with null CustomerID
clean_df = df.dropna(subset = ['CustomerID'])

# Check null values
clean_df.isnull().sum()
# Remove the unspecified countries
clean_df = clean_df[(clean_df.Country != 'Unspecified')]
# Removing the price and quantity that are less than or equal to 0
clean_df = clean_df[(clean_df.Quantity >= 0) & (clean_df.UnitPrice >= 0)]
clean_df.describe()
# Check the number of invoices that starts with letter 'c', cancellation.
clean_df['InvoiceNo'] = clean_df['InvoiceNo'].astype('str')
clean_df[clean_df['InvoiceNo'].str.contains("c")].shape[0]
# Check the stock code

def has_right_scode(input):
    
    """
    Function: check the if the stock code is wirtten in a right way,
            The function check if the code contains 5-digit number or 5-digit number with a letter.
    Args:
      input(String): Stock code
    Return:
      Boolean: True or False
    """
    
    x = re.search("^\d{5}$", input)
    y = re.search("^\d{5}[a-zA-Z]{1}$", input)
    if (x or y):
        return True
    else:
        return False

    
clean_df['StockCode'] = clean_df['StockCode'].astype('str')
clean_df = clean_df[clean_df['StockCode'].apply(has_right_scode) == True]
clean_df.head()
# One discription for each stock code

# Put all Descriptions of each StockCode in a list 
df_itms = pd.DataFrame(clean_df.groupby('StockCode').apply(lambda x: x['Description'].unique())).reset_index()
df_itms.rename(columns = { 0: 'Description2'}, inplace = True)

# StockCode that have more than one Description
df_itms[df_itms['Description2'].str.len() != 1].head()
# Take one Description for each StockCode
df_itms.loc[:, 'Description2'] = df_itms.Description2.map(lambda x: x[0])

# StockCode that have more than one Description
df_itms[df_itms['Description2'].str.len() != 1].head()
# Merge clean_df with df_itms
clean_df = pd.merge(clean_df, df_itms, on = 'StockCode')
clean_df = clean_df.drop('Description', axis = 1)
clean_df.rename(columns = { 'Description2': 'Description'}, inplace = True)
clean_df.head()
# Count number of purchases for each customer
trans_num = clean_df.groupby(['CustomerID'])['InvoiceNo'].count().to_frame().reset_index().rename(columns = {'InvoiceNo': 'Transactions'})
trans_num.head()
# Calculate the median number of purchases
trans_median = round(trans_num["Transactions"].median())
trans_median
# Add the number of the transcations column to the clean data frame
act_cust = pd.merge(clean_df, trans_num, how = 'inner', on = 'CustomerID')

# Keep the customers that have number of transactions >= trans_median 
act_cust = act_cust[act_cust["Transactions"] >= trans_median]
act_cust.head()
# Sum the quantity of each product based on the customer
act_cust = act_cust.groupby(['CustomerID', 'Description']).sum()['Quantity'].to_frame().reset_index()
act_cust.head(10)
# Get the max quantity of products based on customer
act_cust = act_cust.groupby(['CustomerID']).max().reset_index()
act_cust.head()
# List of the preference products for each customer
prf_prod_cust = act_cust[['CustomerID', 'Description']]
prf_prod_cust.head()
# Top 20 favorite products for the customers
prf_prod_cust.groupby(['Description']).count().sort_values('CustomerID', ascending=False).head(20)
# Sum the quantity of each product based on the country
prf_prod_country = clean_df.groupby(['Country', 'Description']).sum()['Quantity'].to_frame().reset_index()
prf_prod_country.head()
# Top 3 favorite products in each country
prf_prod_country = prf_prod_country.set_index('Description').groupby('Country')['Quantity'].nlargest(3).reset_index()
prf_prod_country = prf_prod_country[['Country', 'Description']]
prf_prod_country.head(15)
# Create month column from the InvoiveDate column 
clean_df['Month'] = pd.DatetimeIndex(clean_df['InvoiceDate']).month
clean_df.head()
# Sum the quantity of each product based on the month
prf_prod_mnth = clean_df.groupby(['Month', 'Description']).sum()['Quantity'].to_frame().reset_index()
prf_prod_mnth.head()
# Top 3 favorite products in each country
prf_prod_mnth = prf_prod_mnth.set_index('Description').groupby('Month')['Quantity'].nlargest(3).reset_index()
prf_prod_mnth = prf_prod_mnth[['Month', 'Description']]
prf_prod_mnth
# The fevorite products in more than one month
prf_prod_mnths = prf_prod_mnth.groupby('Description').count() ['Month'].to_frame()
prf_prod_mnths[prf_prod_mnths['Month'] > 1]
prf_prod_mnth_country = clean_df.groupby(['Month', 'Description', 'Country']).sum()['Quantity'].to_frame().reset_index()
prf_prod_mnth_country.head()
# Sum the quantity of each product based on the month and country
prf_prod_mnth_country = prf_prod_mnth_country.set_index('Description').groupby(['Country', 'Month'])['Quantity'].nlargest(3).reset_index()
prf_prod_mnth_country = prf_prod_mnth_country[['Country', 'Month', 'Description']]
prf_prod_mnth_country.head()
