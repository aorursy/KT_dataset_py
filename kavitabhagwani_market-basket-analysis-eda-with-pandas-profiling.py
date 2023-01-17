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
# import required libraries 

import pandas as pd
import numpy as np
import seaborn as sns # used for plot interactive graph. 
import matplotlib.pyplot as plt

#Pandas-profiling is an open source library that can generate beautiful interactive reports for initial EDA.

import pandas_profiling
from pandas_profiling import ProfileReport
# Importing dataset and  checking the first 5 rows of dataset
data= pd.read_csv('../input/online-retail-ii-uci/online_retail_II.csv')
data.head()
# To install pandas profiling in notebook
# pip install pandas-profiling 
# So from below reports can have look for itnital analaysis of missing value, distribution of variables etc..
# Basic analysis reports
ProfileReport(data)
# can check for unique description 
print('Unique Items: ', data['Description'].nunique())
print( '\n', data['Description'].unique())
# CustId and description have null values 
print(data.isnull().sum().sort_values(ascending=False)) 
#drop all null values
data.dropna(inplace = True) 
#No null values are present now
data.info() 
# explore and visualize the most sales items within this time period. 
most_sold = data['Description'].value_counts().head(15)

print('Most Sold Items: \n')
print(most_sold)
#A bar plot of the support of most frequent items bought.
plt.figure(figsize=(7,6))
most_sold.plot(kind='bar')
plt.title('Items Most Sold')
# UK has maximum records of sales followed by Germany
data['Country'].value_counts() 
data = data.loc[data['Country'] == 'Germany']   
data.head()
#remove spaces from begining of the description 
data['Description'] = data['Description'].str.strip()
# drop duplicates of invoices 
data.dropna(axis=0,subset=['Invoice'],inplace = True)
# converting invoice in to string 
data['Invoice'] = data['Invoice'].astype('str')
# Seprating transaction for Germany
my_basket = (data[data['Country']=='Germany']
            .groupby(['Invoice','Description'])['Quantity']
            .sum().unstack().reset_index().fillna(0) 
            .set_index('Invoice'))
my_basket.head() 
# converting all positive value to 1 and rest to zero
def my_encode_units(x):
    if x <= 0:
        return 0
    if x >= 1:
        return 1
my_basket_sets = my_basket.applymap(my_encode_units)
# Import required libraries
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import association_rules, apriori
#Support of at least support 0.7% 

#We sort the rules by decreasing confidence.

frequent_itemsets = apriori(my_basket_sets, min_support=0.07, use_colnames=True)

# generating rules from frequent trasactions 

rules = association_rules(frequent_itemsets, metric='lift', min_threshold=1.0) # different rules having support, confidence,lift
rules.sort_values('confidence', ascending=False)
#How many times the item ROUND SNACK BOXES SET OF 4 FRUITS is occuring:15 line item (ignoring Postage )
my_basket_sets['ROUND SNACK BOXES SET OF 4 FRUITS'].sum()
# 15th line item in consequents col ROUND SNACK BOXES SET OF4 WOODLAND
my_basket_sets['ROUND SNACK BOXES SET OF4 WOODLAND'].sum()
rules[(rules['lift'] >=3) &
    (rules['confidence'] >= 0.3) ]
