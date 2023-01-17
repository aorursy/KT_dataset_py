import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
transactions = pd.read_csv('../input/product-launch-datasets/BusinessCase_Tx.csv')
transactions.head(2)
transactions.info()
t_columns_drop = ['Unnamed: 0', 'description', 'locationRegion', 'locationCity', 'originationDateTime', 'merchantId']
transactions.drop(columns = t_columns_drop, inplace=True)
transactions['categoryTags'].fillna('Unknown', inplace=True)
accounts = pd.read_csv('../input/product-launch-datasets/BusinessCase_Accts.csv')
accounts.head(2)
accounts.info()
a_columns_drop = ['Unnamed: 0', 'branchNumber', 'type', 'openDate', 'iban', 'currency']
accounts.drop(columns=a_columns_drop, inplace=True)
customers = pd.read_csv('../input/product-launch-datasets/BusinessCase_Custs.csv')
customers.head(2)
customers.info()
c_columns_drop = ['Unnamed: 0', 'type', 'occupationIndustry', 'habitationStatus',\
                  'addresses_principalResidence_province', 'schoolAttendance', 'schools']
customers.drop(columns=c_columns_drop, inplace=True)
customers['workActivity'].fillna('Unknown', inplace=True)
customers['birthDate'] = pd.to_datetime(customers['birthDate'])
transactions_count = transactions.groupby(['categoryTags'])['customerId'].count().to_frame().sort_values('customerId', ascending=False).reset_index()
transactions_count = transactions_count.rename(columns={'categoryTags': 'Category', 'customerId': 'Total Transactions'})
transactions_sum = transactions.groupby(['categoryTags'])['currencyAmount'].sum().to_frame().sort_values('currencyAmount', ascending=False).reset_index()
transactions_sum = transactions_sum.rename(columns={'categoryTags': 'Category', 'currencyAmount': 'Total Revenue'})
transactions_joined = pd.merge(transactions_count, transactions_sum, on='Category')
transactions_joined.to_csv('transactions_joined.csv')
transactions_joined
total_transactions = transactions_joined['Total Transactions']
sum_transactions = transactions_joined['Total Revenue']
labels = transactions_joined['Category']

ax = plt.figure(figsize=(16,6))
ax = sns.scatterplot(sum_transactions, total_transactions)
plt.xlabel('Total Revenue')
plt.ylabel('Total Transactions')

## This is a function for annotating the labels on the scatterplot ##

def label_point(x, y, val, ax):
    a = pd.concat({'x': x, 'y': y, 'val': val}, axis=1)
    for i, point in a.iterrows():
        ax.text(point['x']+.02, point['y'], str(point['val']))

label_point(sum_transactions, total_transactions, labels, plt.gca())
joined_table = transactions.merge(customers, left_on='customerId', right_on='id')
final_table = joined_table.loc[joined_table['categoryTags'] == 'Transfer'].reset_index(drop=True)
final_table.drop(columns=['accountId', 'categoryTags'], inplace=True)
final_table.head(3)