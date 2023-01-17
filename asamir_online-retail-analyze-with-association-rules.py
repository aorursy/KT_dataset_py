# Importing the libraries

import numpy as np

import matplotlib.pyplot as plt

import pandas as pd
# Importing the dataset

dataset = pd.read_excel('../input/Online Retail.xlsx')

dataset.head()
# Add extra fields 

dataset['TotalAmount'] = dataset['Quantity'] * dataset['UnitPrice']

dataset['InvoiceYear'] = dataset['InvoiceDate'].dt.year

dataset['InvoiceMonth'] = dataset['InvoiceDate'].dt.month

dataset['InvoiceYearMonth'] = dataset['InvoiceYear'].map(str) + "-" + dataset['InvoiceMonth'].map(str)
## Part1. Exploring

######################

####################### Explore In General ###############

dataset.describe()
# Total number of transactions

len(dataset['InvoiceNo'].unique())
# Number of transactions with anonymous customers 

len(dataset[dataset['CustomerID'].isnull()]['InvoiceNo'].unique())
# Total numbers of customers - +1 for null users

len(dataset['CustomerID'].unique())
# Total profit

sum(dataset['TotalAmount'])
# Get top ranked ranked customers based on the total amount

customers_amounts = dataset.groupby('CustomerID')['TotalAmount'].agg(np.sum).sort_values(ascending=False)

customers_amounts.head(20)
customers_amounts.head(20).plot.bar()
# Frequently sold items by quantitiy

gp_stockcode = dataset.groupby('Description')

gp_stockcode_frq_quantitiy = gp_stockcode['Quantity'].agg(np.sum).sort_values(ascending=False)

gp_stockcode_frq_quantitiy.head(20)
gp_stockcode_frq_quantitiy.head(20).plot.bar()
# Frequently sold items by total amount

gp_stockcode_frq_amount = gp_stockcode['TotalAmount'].agg(np.sum).sort_values(ascending=False)

gp_stockcode_frq_amount.head(20)
gp_stockcode_frq_amount.head(20).plot.bar()
# Explore by month

gp_month = dataset.sort_values('InvoiceDate').groupby(['InvoiceYear', 'InvoiceMonth'])
# Month number of invoices

gp_month_invoices = gp_month['InvoiceNo'].unique().agg(np.size)

gp_month_invoices
gp_month_invoices.plot.bar()
# Month total amounts

gp_month_frq_amount= gp_month['TotalAmount'].agg(np.sum)

gp_month_frq_amount
gp_month_frq_amount.plot.bar()
####################### Explore By Countries ###############

gp_country = dataset.groupby('Country')

# Order countries by total amount

gp_country['TotalAmount'].agg(np.sum).sort_values(ascending=False)
# Order countries by number of invoices

gp_country['InvoiceNo'].unique().agg(np.size).sort_values(ascending=False)
# Order countries by number of customers

gp_country['CustomerID'].unique().agg(np.size).sort_values(ascending=False)
# Work on undefined customers

gp_country_null = dataset[dataset['CustomerID'].isnull()].groupby('Country')

# Order countries by total amount [For the undefined users]

gp_country_null['TotalAmount'].agg(np.sum).sort_values(ascending=False)
# Order countries by number of invoices [For the undefined users]

gp_country_null['InvoiceNo'].unique().agg(np.size).sort_values(ascending=False)
# Explore more info about United Kingdom invoices because it has the max total amount

# Get United Kingdom top ranked customers based on the total amount

uk_customers_amounts = dataset[dataset['Country']=='United Kingdom'].groupby('CustomerID')['TotalAmount'].agg(np.sum).sort_values(ascending=False)

uk_customers_amounts.head(20)
uk_customers_amounts.head(20).plot.bar()
# United Kingdom frequently sold items by quantitiy

uk_gp_stockcode = dataset[dataset['Country']=='United Kingdom'].groupby('Description')

uk_gp_stockcode_frq_quantitiy = uk_gp_stockcode['Quantity'].agg(np.sum).sort_values(ascending=False)

uk_gp_stockcode_frq_quantitiy.head(20)
uk_gp_stockcode_frq_quantitiy.head(20).plot.bar()
# Frequently sold items by total amount

uk_gp_stockcode_frq_amount = uk_gp_stockcode['TotalAmount'].agg(np.sum).sort_values(ascending=False)

uk_gp_stockcode_frq_amount.head(20)
uk_gp_stockcode_frq_amount.head(20).plot.bar()
# Explore United Kingdom by month

uk_gp_month = dataset[dataset['Country']=='United Kingdom'].groupby(['InvoiceYear', 'InvoiceMonth'])

# United Kingdom By Month number of invoices

uk_gp_month_invoices = uk_gp_month['InvoiceNo'].unique().agg(np.size)

uk_gp_month_invoices
uk_gp_month_invoices.plot.bar()
# United Kingdom By Month total amounts

uk_gp_month_frq_amount= uk_gp_month['TotalAmount'].agg(np.sum)

uk_gp_month_frq_amount
uk_gp_month_frq_amount.plot.bar()
## Part2. Get Association Rules

#################################

#set null description = stockCode

len(dataset[dataset['Description'].isnull()])
for i, d in dataset[dataset['Description'].isnull()].iterrows():

    dataset['Description'][i] = "Code-" + str(d['StockCode'])
len(dataset[dataset['Description']==dataset['StockCode'].map(lambda x: "Code-"+str(x))])
# Set transactions

gp_invoiceno = dataset.groupby('InvoiceNo')

transactions = []

for name,group in gp_invoiceno:

    transactions.append(list(group['Description'].map(str)))
# Training Apriori on the dataset

# Needs to import https://pypi.python.org/pypi/apyori/1.0.0

from apyori import apriori

rules = apriori(transactions, min_support = 0.005, min_confidence = 0.2, min_lift = 3, min_length = 2)

# Get the results

results = list(rules)
# Visualising the result as a readeable dataframe

final_results = pd.DataFrame(np.random.randint(low=0, high=1, size=(len(results), 6)), columns=['GeneralRules', 'LeftRules', 'RightRules', 'Support', 'Confidence', 'Lift'])

index = 0

for g, s, i in results:

    final_results.iloc[index] = [' _&&_ '.join(list(g)), ' _&&_ '.join(list(i[0][0])), ' _&&_ '.join(list(i[0][1])), s, i[0][2], i[0][3]]

    index = index+1

# The most significant rules

final_results = final_results.sort_values('Lift', ascending=0)

final_results.head(20)
count=1

for i, d in final_results.head(20).iterrows():

    print('Rule #'+str(count)+':')

    print(d['LeftRules'])

    print('=> '+d['RightRules'])

    print('Support: '+str(d['Support'])+' - Confidence: '+str(d['Confidence'])+' - Lift: '+str(d['Lift']))

    print('--------------------')

    count=count+1