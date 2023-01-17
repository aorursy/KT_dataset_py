# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

from PIL import Image

from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

import datetime as dt

from datetime import timedelta



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/data.csv", encoding="ISO-8859-1")
df.info()
df.isnull().sum()/df.shape[0]
df.dropna(subset=['CustomerID'], inplace=True)
df.head(5)
print('Number of rows', df.shape[0])
#types

for c in ['InvoiceNo', 'StockCode', 'Description', 'CustomerID', 'Country']:

    df[c] = df[c].astype('category')

#dates    

df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

df['Year'] = df['InvoiceDate'].dt.year

df['Month'] = df['InvoiceDate'].dt.month

df['Day'] = df['InvoiceDate'].dt.day

df['DayOfWeek'] = df['InvoiceDate'].dt.dayofweek

df['DayOfWeek'] = df['DayOfWeek'].map({0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday', 4: 'Friday', 5: 'Saturday', 6: 'Sunday'})

df['Hour'] = df['InvoiceDate'].dt.hour



#invoice

df['StockValue'] = df['Quantity']*df['UnitPrice']
print(df['InvoiceDate'].min())

print(df['InvoiceDate'].max())

print(df['InvoiceDate'].max() - df['InvoiceDate'].min())
def column_statistic(column):

    plt.figure(figsize=(20,10))

    sns.boxplot(df[column])

    print(df[column].describe())

    print('-'*10)

    print('Variation coefficient {}'.format(df[column].std()/df[column].mean()))

    print('-'*10)

    print('Minimum values')

    print(df[column].sort_values().head(5))

    print('Maximum values')

    print(df[column].sort_values(ascending=False).head(5))
column_statistic('Quantity')
df[df['Quantity'] < 0].sample(5)
df['InvoiceType'] = df['InvoiceNo'].apply(lambda x: 'FK' if x[:1] == 'C' else 'FV')
column_statistic('UnitPrice')
df[df['UnitPrice'] == 0].sample(5)
group_type = df.groupby('InvoiceType')['InvoiceNo'].nunique()

print('We have got {0} invoices and {1} correcting invoices'.format(group_type[1],

                                                                    group_type[0]))

print('Its about {0} invoices and {1} correcting invoices per day'.format(group_type[1]/373,

                                                                        group_type[0]/373))
date_group = df.groupby('Month')['Day'].nunique()

plt.figure(figsize=(20,10))

sns.barplot(date_group.index, date_group.values)

plt.ylabel('Number of working days')

plt.xlabel('Month')

plt.title('Number of days in each month that transactions were made', fontsize=15)

plt.show()
dayweek_group = df.groupby('DayOfWeek')['InvoiceNo'].nunique()

plt.figure(figsize=(20,10))

sns.barplot(dayweek_group.index, dayweek_group.values, order=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Sunday'])

plt.ylabel('Number of invoices')

plt.xlabel('Day')

plt.title('Number of invoices for every day', fontsize=15)

plt.show()
index = df['CustomerID'].drop_duplicates().index

df_unique_customer = df.loc[index, :]

unique_customer_count = df_unique_customer['Country'].value_counts()

#############

plt.figure(figsize=(20,15))

sns.barplot(y=unique_customer_count.index, x=unique_customer_count.values,

            order=unique_customer_count.index)

plt.yticks(size=15)

plt.ylabel('Country', fontsize=15)

plt.xlabel('Invoices', fontsize=15)

plt.title('Number of invoices per each country', fontsize=15)

print('There are {} customers who made {} transactions'.format(len(df['CustomerID'].unique()),

                                                             len(df['InvoiceNo'].unique())))
##data

customer_value = df.groupby('CustomerID')['StockValue'].sum().sort_values(ascending=False)

##plot

plt.figure(figsize=(20,15))

plt.barh(y=np.linspace(0,19,20), width=customer_value[:20][::-1].values, align='center', linewidth=10)

plt.yticks(np.linspace(0,19,20), customer_value[:20][::-1].index, size=15)

plt.title('Most profitable customers', fontsize=15)

plt.xlabel('Stock value', fontsize=15)

plt.ylabel('Customer ID', fontsize=15)

plt.show()
customer_20 = customer_value[:20].reset_index()



for i, customer in enumerate(customer_20['CustomerID']):

    value_for_customer = df[df['CustomerID'] == customer]['Country'].unique()[0]

    customer_20.loc[i, 'Country'] = value_for_customer

print('Top 20 most profitable customers are from: ')

customer_20['Country'].value_counts()

                                       
stock_code_count = len(df['StockCode'].unique())

print('There are {} different stock codes'.format(stock_code_count))
text = " ".join(review for review in df['Description'])

wordcloud = WordCloud(background_color="white").generate(text)

plt.figure(figsize=(15,15))

plt.imshow(wordcloud, interpolation='bilinear')

plt.axis("off")

plt.title('Most frequent words in description of item', fontsize=15)

plt.show()
#stock_code_v_counts = df['StockCode'].value_counts(normalize=True)[:20]

stock_code_v_counts = df.groupby('StockCode')['StockValue'].sum().sort_values(ascending=False)[:20]

stock_code_v_counts = stock_code_v_counts/df['StockValue'].sum()

plt.figure(figsize=(20,10))

sns.barplot(y=stock_code_v_counts.index, x=stock_code_v_counts.values,

            order=stock_code_v_counts.index, color='blue')

plt.title('Most profitable stock codes', fontsize=15)

plt.xlabel('Percent of total stock value', fontsize=15)

plt.ylabel('Stock code', fontsize=15)

plt.yticks(size=15)

plt.show()
description_stock_code = stock_code_v_counts.reset_index()

for i, stockcode in enumerate(description_stock_code['StockCode']):

    value_for_stockcode = df[df['StockCode'] == stockcode]['Description'].value_counts().index[0]

    description_stock_code.loc[i, 'Description'] = value_for_stockcode

print('What are these 20 top profitable stock codes?')    

description_stock_code['Description']

digit = df['StockCode'].apply(lambda x: 'char' if x.isalpha() else 'number')

char_index = digit[digit == 'char'].index



df_char_stock = df.loc[char_index, :]

print('Unusual stock codes: {}'.format([x for x in df_char_stock['StockCode'].unique()])) 
for stockcode in df_char_stock['StockCode'].unique():

    print('Stock code: {} has {} unique values  - Description {}'.format(stockcode,

                                                                        df[df['StockCode'] == stockcode]['Description'].nunique(),

                                                                       df[df['StockCode'] == stockcode]['Description'].unique()[0]))

    
df_low = df[df['InvoiceType'] == 'FV']

outlier_drop = df_low['StockValue'].quantile(0.95)

df_low = df_low[df_low['StockValue'] < outlier_drop]

print(df_low.shape[0]/df.shape[0])
df_low['CustomerID'].nunique()/df['CustomerID'].nunique()
stock_value_group_mean = df_low['StockValue'].groupby(df_low['Month']).mean()

stock_value_group_std = df_low['StockValue'].groupby(df_low['Month']).std()

plt.figure(figsize=(20,10))

plt.plot(stock_value_group_mean, marker="o", label='Mean stock value')

plt.fill_between(stock_value_group_mean.index, stock_value_group_mean.values - stock_value_group_std.values, stock_value_group_mean.values + stock_value_group_std.values,

                 color='gray', alpha=0.4, label='Standard deviation')

plt.xticks(np.linspace(1,12,12))

plt.xlabel('Month', fontsize=15)

plt.ylabel('Stock value', fontsize=15)

plt.legend()

plt.show()
df_low['StockValue'].std()/df_low['StockValue'].mean()
date = df_low['InvoiceDate'].max() + timedelta(days=1)
rfm = df_low.groupby(['CustomerID']).agg({'StockValue': lambda x: x.sum(),

                                'CustomerID': 'count',

                               'InvoiceDate': lambda x: (date-x.max()).days})

rfm.rename(columns={'StockValue': 'TotalMonetary',

                    'CustomerID': 'TotalTransactions',

                    'InvoiceDate': 'LastPurchase'}, inplace=True)

rfm.dropna(inplace=True)





print(rfm.shape[0])

rfm.head(5)
quantile = rfm.quantile(np.linspace(0,1,5))

quantile
def get_bins_monetary(x):

    if x <= quantile.loc[0, 'TotalMonetary']:

        return 5

    elif x <= quantile.loc[0.25, 'TotalMonetary']:

        return 4

    elif x <= quantile.loc[0.5, 'TotalMonetary']:

        return 3

    elif x <= quantile.loc[0.75, 'TotalMonetary']:

        return 2

    else:

        return 1

    

def get_bins_transactions(x):

    if x <= quantile.loc[0, 'TotalTransactions']:

        return 5

    elif x <= quantile.loc[0.25, 'TotalTransactions']:

        return 4

    elif x <= quantile.loc[0.5, 'TotalTransactions']:

        return 3

    elif x <= quantile.loc[0.75, 'TotalTransactions']:

        return 2

    else:

        return 1

    

def get_bins_purchase(x):

    if x <= quantile.loc[0, 'LastPurchase']:

        return 1

    elif x <= quantile.loc[0.25, 'LastPurchase']:

        return 2

    elif x <= quantile.loc[0.5, 'LastPurchase']:

        return 3

    elif x <= quantile.loc[0.75, 'LastPurchase']:

        return 4

    else:

        return 5
rfm['m_rate'] = rfm['TotalMonetary'].apply(get_bins_monetary)

rfm['f_rate'] = rfm['TotalTransactions'].apply(get_bins_transactions)

rfm['r_rate'] = rfm['LastPurchase'].apply(get_bins_purchase)

rfm['RFM'] = rfm['m_rate'].map(str) + rfm['f_rate'].map(str) + rfm['r_rate'].map(str)

rfm['RFM_value'] = rfm['m_rate'] + rfm['f_rate'] + rfm['r_rate']

rfm.head(10)
cross_table1 = pd.crosstab(index=rfm['m_rate'], columns=rfm['f_rate'])

cross_table2 = pd.crosstab(index=rfm['m_rate'], columns=rfm['r_rate'])

cross_table3 = pd.crosstab(index=rfm['f_rate'], columns=rfm['r_rate'])

plt.figure(figsize=(20,15))

plt.subplot(311)

ax1 = sns.heatmap(cross_table1, cmap='viridis', annot=True, fmt=".0f")

ax1.invert_yaxis()

ax1.set_ylabel('Monetary')

ax1.set_xlabel('Frequency')

plt.subplot(312)

ax2 = sns.heatmap(cross_table2, cmap='viridis', annot=True, fmt=".0f")

ax2.invert_yaxis()

ax2.set_ylabel('Monetary')

ax2.set_xlabel('Recency')

plt.subplot(313)

ax3 = sns.heatmap(cross_table3, cmap='viridis', annot=True, fmt=".0f")

ax3.invert_yaxis()

ax3.set_ylabel('Frequency')

ax3.set_xlabel('Recency')

plt.show()
print('Correlation monetary - frequency: ', rfm[['m_rate', 'f_rate']].corr().iloc[1,0])

print('Correlation monetary - recency: ', rfm[['m_rate', 'r_rate']].corr().iloc[1,0])

print('Correlation frequency - recency: ', rfm[['f_rate', 'r_rate']].corr().iloc[1,0])