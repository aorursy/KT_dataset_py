# Importing Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
import os
sns.set()
%matplotlib inline
df= []
for file in os.listdir('../input/'):
    command = file[:-4] + ' = pd.read_csv(\'../input/'  + file + '\')'
    exec(command)
    df.append(file[:-4])
print(df)
order_products_prior = order_products__prior.copy()
order_products_train = order_products__train.copy()
orders.head()
orders.describe()
orders['eval_set'].value_counts()
orders.groupby('eval_set')['user_id'].apply(lambda x: len(np.unique(x)))
order_counts_by_user = orders.groupby('user_id')['order_number'].count().reset_index()['order_number'].value_counts()
plt.figure(figsize=(24,12))
sns.barplot(order_counts_by_user.index, order_counts_by_user.values)

plt.ylabel('Counts', fontsize=20)
plt.xlabel('Number of Orders', fontsize=20)
plt.xticks(rotation=90, fontsize=16)
plt.show()
#Add a new column with weekday for plots
orders['Weekday'] = orders['order_dow'].map({1:'Sun', 2:'Mon', 3:'Tue', 4:'Wed', 5:'Thu',6:'Fri',0:'Sat'})

# Plot to show how orders are distributed over the days of week
plt.figure(figsize=(12,8))
sns.countplot('Weekday', data=orders.sort_values('order_dow'), color='green')
plt.ylabel('Counts', fontsize=22)
plt.xlabel('Day of Week', fontsize=22)
plt.title('Number of Orders vs Day of Week', fontsize=30)
plt.yticks(fontsize=16)
plt.xticks(rotation=90, fontsize=16)
plt.show()
# Plot to show how orders are distributed over the hours of day
plt.figure(figsize=(12,10))
sns.countplot('order_hour_of_day', data=orders, color='blue',alpha=0.5)
plt.ylabel('Counts', fontsize=22)
plt.xlabel('Hour of Day', fontsize=22)
plt.title('Number of Orders vs Hour of Day', fontsize=30)
plt.yticks(fontsize=16)
plt.xticks(rotation=90, fontsize=16)
plt.axvline(10, color='red', linewidth=3)
plt.axvline(16, color='red', linewidth=3)
plt.show()
grouped = orders.groupby(['Weekday', 'order_hour_of_day'])['order_number'].aggregate('count').reset_index()
grouped = grouped.pivot('Weekday', 'order_hour_of_day', 'order_number')
plt.figure(figsize=(12,10))
sns.heatmap(grouped, cmap='BuGn' )
plt.xlabel('Hour of Day', fontsize=16)
plt.ylabel('Day of Week', fontsize=16)
plt.yticks(rotation=0)
plt.show()
plt.figure(figsize=(12,8))
sns.countplot(orders['days_since_prior_order'],color='purple', alpha=0.3)
plt.ylabel('Counts', fontsize=16)
plt.xlabel('Days Since Previous Order', fontsize=16)
plt.xticks(rotation=90)
plt.show()
aisles.head()
products.head()
departments.head()
order_products_prior.head()
# order_products_prior = pd.merge(order_products_prior, products, on='product_id', how='left')
# order_products_prior = pd.merge(order_products_prior, aisles, on='aisle_id', how='left')
# order_products_prior = pd.merge(order_products_prior, departments, on='department_id', how='left')
# order_products_prior.head()


def merge(df):
    df = pd.merge(df, products, on='product_id', how='left')
    df = pd.merge(df, aisles, on='aisle_id', how='left')
    df = pd.merge(df, departments, on='department_id', how='left')
    return df

order_products_prior = merge(order_products_prior)
order_products_train = merge(order_products_train)
def high_frequency_plot(col):
    plt.figure(figsize=(14,10))
    order_products_prior[col].value_counts().sort_values(ascending=False).head(25).plot(kind='bar')
    plt.title('Frequency distribution', fontsize=20)
    plt.xticks(fontsize=18)
    plt.ylabel('Counts', fontsize=18)
    plt.show()
high_frequency_plot('product_name')
high_frequency_plot('aisle')
high_frequency_plot('department')
temp_df = pd.DataFrame(order_products_prior['department'].value_counts().sort_values(ascending=False)).head(10)
temp_df['fraction_of_total'] = temp_df['department']/temp_df['department'].sum()*100
plt.figure(figsize=(10,10))
plt.pie(temp_df['fraction_of_total'], labels=temp_df.index, autopct='%1.1f%%')
plt.title('Department Distribution', fontsize=18)
plt.show()
temp_df = order_products_prior.groupby('product_name')['reordered'].mean().sort_values(ascending=False)

plt.figure(figsize=(12,10))
sns.pointplot(temp_df.head(20).index,temp_df.head(20))
plt.xlabel('Products', fontsize=18)
plt.ylabel('Reorder Ratio', fontsize=(18))
plt.xticks(rotation=90,fontsize=12)
plt.show()

print('Number of products never reordered = ', (temp_df==0.0).sum())
temp_df = order_products_prior.groupby('aisle')['reordered'].mean().sort_values(ascending=False)

plt.figure(figsize=(12,10))
sns.pointplot(temp_df.head(50).index,temp_df.head(50))
plt.xlabel('Aisles', fontsize=18)
plt.ylabel('Reorder Ratio', fontsize=(18))
plt.xticks(rotation=90,fontsize=12)
plt.show()

temp_df = order_products_prior.groupby('department')['reordered'].mean().sort_values(ascending=False)

plt.figure(figsize=(12,10))
sns.pointplot(temp_df.index,temp_df)
plt.xlabel('department', fontsize=18)
plt.ylabel('Reorder Ratio', fontsize=(18))
plt.xticks(rotation=90,fontsize=12)
plt.show()