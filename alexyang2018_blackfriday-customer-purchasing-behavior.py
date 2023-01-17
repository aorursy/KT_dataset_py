# To explore customer purchase behavior using transaction data from a retail store on Black Friday
# 1. what are the demands for different product categories?
# 2. which product categories generate the most revenue?
# 3. who purchased more?
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
%matplotlib inline

import seaborn as sns
import scipy.stats

import os

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
bf = pd.read_csv('../input/BlackFriday.csv')
# Quick check
bf.info()
bf.sort_values('User_ID').head(10)
bf['User_ID'].value_counts().count()   #5,891 customers
# Get to know data - Product_ID, Product_Category*
# 3,623 distinct products, grouped into 18 main categories (i.e. Product_Category_1)
# Drop product category 2 & 3 - lots of missing values 
# Drop Product_ID - may not be very informative to look at individual product
bf['Product_ID'].value_counts().count()
bf['Product_Category_1'].value_counts(dropna = False).sort_index()
bf['Product_Category_2'].value_counts(dropna = False).sort_index()
bf['Product_Category_3'].value_counts(dropna = False).sort_index()

bf = bf.drop(['Product_ID', 'Product_Category_2', 'Product_Category_3'], axis = 1)
# Generate new features - total # products purchased by customer; and total amount 
tot_item = bf['User_ID'].value_counts().sort_index()
tot_purchase = bf.groupby('User_ID').sum()['Purchase']
tot = pd.concat([tot_item, tot_purchase], axis = 1, keys = ['Tot_Products', 'Tot_Purchase'])

bf = pd.merge(bf, tot, left_on = 'User_ID', right_index = True)

bf.head()
# Data Exploration
#1) Demand & revenue by product category:
# 3 best sellers are product category 5, 1, 8. While product category 1 generated a lot more revenue than 5 & 8,
# about twice as much - could potentially allocate more resource to product category 5, 1 & 8, especially category
# 1, e.g. more stock, for next Black Friday
fig1, axes = plt.subplots(2, 1, figsize = (10, 6))

fig1.suptitle('Demand & Revenue by Product Category', fontsize = 16, y = 0.95)

demand_product_cat = bf['Product_Category_1'].value_counts().sort_index()
demand_product_cat.plot(kind = 'bar', ax = axes[0])
plt.sca(fig1.axes[0])
plt.ylabel('Quantity')

revenue_product_cat = bf.groupby('Product_Category_1').sum()['Purchase']/1000000 # $(in million)
revenue_product_cat.plot(kind = 'bar', ax = axes[1])
plt.sca(fig1.axes[1])
plt.xlabel('Product Category')
plt.ylabel('Revenue (/Million $)')

plt.savefig('fig1.png')
#2) Who purchased the most to each product category? Or which products were more attractive to which customers?
chars = ['Gender', 'Age', 'Occupation', 'City_Category', 'Stay_In_Current_City_Years', 'Marital_Status']

fig2, axes = plt.subplots(3, 2, figsize = (20, 16))

fig2.suptitle('Product Demand by Customer Groups', fontsize = 16, y = 0.92)

n = 0

for i in range(3):
    for j in range(2):
        p = pd.crosstab(bf['Product_Category_1'], bf[chars[n]])
        p = p.apply(lambda r: r/r.sum(), axis = 1)
        p.plot(kind = 'bar', ax = axes[i][j], stacked = True)
        
        n += 1
        
for ax in fig2.axes:
    plt.sca(ax)
    plt.xlabel('Product Category')
    

plt.savefig('fig2.png')
#4) Customer's purchase behavior - who purchased more?
# Construct customer level data
bf_customer = bf.copy()
bf_customer.drop(['Product_Category_1', 'Purchase'], axis = 1, inplace = True)

bf_customer.drop_duplicates(inplace = True)
bf_customer['Tot_Purchase'] = bf_customer['Tot_Purchase']/1000    # per thousand $
bf_customer.count()['User_ID']   #5,891 customers
bf_customer.head()
# Look at the distribution of total amount of purchase by customer - right skewed
fig4 = plt.figure(figsize = (10, 6))
sns.distplot(bf_customer['Tot_Purchase'],kde = False)

plt.title('Distribution of Total Amount of Purchase by Customer')
plt.xlabel('Total Amount of Purchase(/Thousand $)')
plt.ylabel('# Customers')

plt.savefig('fig4.png')
# Look at correlations between total amount of purchase and characteristics of customer
for var in chars:
    bf_customer.groupby(var).describe()['Tot_Purchase']
# Bar charts - show median instead of mean of total amount of purchase by each characteristic
fig5, axes = plt.subplots(3,2,figsize=(20,16))

fig5.suptitle('Median Amount of Purchase by Customer Groups', fontsize = 16, y = 0.93)

sns.barplot(x='Gender', y='Tot_Purchase', data = bf_customer, estimator = np.median, ci = None, ax = axes[0][0])
sns.barplot(x='Age', y='Tot_Purchase', data = bf_customer, estimator = np.median, ci = None, 
            ax = axes[0][1], order = ['0-17', '18-25', '26-35', '36-45', '46-50', '51-55', '55+'])
sns.barplot(x='Occupation', y='Tot_Purchase', data = bf_customer, estimator = np.median, ci = None, ax = axes[1][0])
sns.barplot(x='City_Category', y='Tot_Purchase', data = bf_customer, estimator = np.median, 
            ci = None, ax = axes[1][1], order = ('A', 'B', 'C'))
sns.barplot(x='Stay_In_Current_City_Years', y='Tot_Purchase', data = bf_customer, estimator = np.median, 
            ci = None, ax = axes[2][0], order = ('0', '1', '2', '3', '4+'))
sns.barplot(x='Marital_Status', y='Tot_Purchase', data = bf_customer, estimator = np.median, ci = None, ax = axes[2][1])

for ax in fig5.axes:
    plt.sca(ax)
    plt.ylabel('Median Total Amount of Purchase (/Thousand $)')
 
plt.savefig('fig5')
