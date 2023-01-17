# Import needed libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from mlxtend.frequent_patterns import apriori, association_rules
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
df.groupby('StockCode').apply(lambda x: x['Description'].unique())
# Number of invoices for each country
df.groupby(['Country']).count() ['InvoiceNo']
# Delete rows with null CustomerID
clean_df = df.dropna(subset = ['CustomerID'])

# Check null values
clean_df.isnull().sum()
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
df_itms[df_itms['Description2'].str.len() != 1]
# Take one Description for each StockCode
df_itms.loc[:, 'Description2'] = df_itms.Description2.map(lambda x: x[0])

# StockCode that have more than one Description
df_itms[df_itms['Description2'].str.len() != 1]
# Merge clean_df with df_itms
clean_df = pd.merge(clean_df, df_itms, on = 'StockCode')
clean_df = clean_df.drop('Description', axis = 1)
clean_df.head()
clean_df.rename(columns = { 'Description2': 'Description'}, inplace = True)
clean_df.head()
df_itms_togthr = clean_df.groupby(['InvoiceNo','Description'])['Quantity'].sum()
df_itms_togthr.head()
df_itms_togthr = df_itms_togthr.unstack().fillna(0)
df_itms_togthr.head()
# Encode the frequency of description to 0 or 1
encode = lambda x : 1 if (x >= 1) else 0
df_itms_togthr = df_itms_togthr.applymap(encode)
df_itms_togthr.head()
df_itms_togthr.shape
nl_df = clean_df[clean_df['Country'] == 'Netherlands']
spain_df = clean_df[clean_df['Country'] == 'Spain']
france_df = clean_df[clean_df['Country'] == 'France']
nl_df.head()
spain_df.head()
france_df.head()
df_itms_togthr_nl = nl_df.groupby(['InvoiceNo','Description'])['Quantity'].sum()

df_itms_togthr_nl = df_itms_togthr_nl.unstack().fillna(0)

encode = lambda x : 1 if (x >= 1) else 0
df_itms_togthr_nl = df_itms_togthr_nl.applymap(encode)
df_itms_togthr_nl.head()
df_itms_togthr_spain = spain_df.groupby(['InvoiceNo','Description'])['Quantity'].sum()

df_itms_togthr_spain = df_itms_togthr_spain.unstack().fillna(0)

encode = lambda x : 1 if (x >= 1) else 0
df_itms_togthr_spain = df_itms_togthr_spain.applymap(encode)
df_itms_togthr_spain.head()
df_itms_togthr_france = france_df.groupby(['InvoiceNo','Description'])['Quantity'].sum()

df_itms_togthr_france = df_itms_togthr_france.unstack().fillna(0)

encode = lambda x : 1 if (x >= 1) else 0
df_itms_togthr_france = df_itms_togthr_france.applymap(encode)
df_itms_togthr_france.head()
# Build the Apriori model
rep_items = apriori(df_itms_togthr, min_support = 0.02, use_colnames = True, verbose = 1)
rep_items.head()
# Generate the association rules dataframe
rules = association_rules(rep_items, metric = "confidence", min_threshold = 0.6)
rules
# The number of rules
rules.shape[0]
# Show confidence distribution
plt.hist(rules['confidence'])
plt.show()
# Show the rules that have confidance > 0.75
high_confidance = rules[rules['confidence'] > 0.75]
high_confidance [['antecedents', 'consequents']]
# Build the Apriori model
rep_items_nl = apriori(df_itms_togthr_nl, min_support = 0.1, use_colnames = True, verbose = 1)
rep_items_nl.head()
# Generate the association rules dataframe
rules_nl = association_rules(rep_items_nl, metric = "confidence", min_threshold = 0.6)
rules_nl.head()
# The number of rules
rules_nl.shape[0]
high_confidance_nl = rules_nl[rules_nl['confidence'] > 0.75]
high_confidance_nl [['antecedents', 'consequents']]
# Build the Apriori model
rep_items_spain = apriori(df_itms_togthr_spain, min_support = 0.1, use_colnames = True, verbose = 1)
rep_items_spain.head()
# Generate the association rules dataframe
rules_spain = association_rules(rep_items_spain, metric = "confidence", min_threshold = 0.6)
rules_spain
# The number of rules
rules_spain.shape[0]
high_confidance_spain = rules_spain[rules_spain['confidence'] > 0.75]
high_confidance_spain [['antecedents', 'consequents']]
# Build the Apriori model
rep_items_france = apriori(df_itms_togthr_france, min_support = 0.05, use_colnames = True, verbose = 1)
rep_items_france.head()
# Generate the association rules dataframe
rules_france = association_rules(rep_items_france, metric = "confidence", min_threshold = 0.6)
rules_france
# The number of rules
rules_france.shape[0]
high_confidance_france = rules_france[rules_france['confidence'] > 0.75]
high_confidance_france [['antecedents', 'consequents']]