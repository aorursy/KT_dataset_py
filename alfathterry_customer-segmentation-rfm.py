import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
%matplotlib inline

!pip install lifetimes

!pip install jcopml
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from lifetimes.utils import summary_data_from_transaction_data

from jcopml.plot import plot_missing_value
df = pd.read_csv("/kaggle/input/onlineretail/OnlineRetail.csv",error_bad_lines=False, encoding='unicode_escape')

df.head()
df.dtypes
plot_missing_value(df, return_df = True)
df.describe()
q1_quan = df['Quantity'].quantile(0.25)

q3_quan = df['Quantity'].quantile(0.75)

iqr_quan = q3_quan - q1_quan

lb_quan = float(q1_quan) - (1.5 * iqr_quan)

ub_quan = float(q3_quan) + (1.5 * iqr_quan)



print('Q1 = {}'.format(q1_quan))

print('Q3 = {}'.format(q3_quan))

print('IQR = Q3 - Q1 = {}'.format(iqr_quan))

print('lower bound = Q1 - 1.5 * IQR = {}'.format(lb_quan))

print('upper bound = Q3 + 1.5 * IQR = {}'.format(ub_quan))
q1_unit = df['UnitPrice'].quantile(0.25)

q3_unit = df['UnitPrice'].quantile(0.75)

iqr_unit = q3_unit - q1_unit 

lb_unit = float(q1_unit) - (1.5 * iqr_unit)

ub_unit = float(q3_unit) + (1.5 * iqr_unit)



print('Q1 = {}'.format(q1_unit))

print('Q3 = {}'.format(q3_unit))

print('IQR = Q3 - Q1 = {}'.format(iqr_unit))

print('lower bound = Q1 - 1.5 * IQR = {}'.format(lb_unit))

print('upper bound = Q1 - 1.5 * IQR = {}'.format(ub_unit))
sns.scatterplot(df['UnitPrice'], df['Quantity'])

plt.title('Quantity x UnitPrice', fontsize = 20);
dx = df[df['Quantity']>0] #hilangkan value negatif

dy = df[df['UnitPrice']>0] #hilangkan value negatif



filtered_quantity = dx.query('(@q1_quan - 1.5 * @iqr_quan) <= Quantity <= (@q3_quan + 1.5 * @iqr_quan)')

filtered_unitprice = dy.query('(@q1_unit - 1.5 * @iqr_unit) <= UnitPrice <= (@q3_unit + 1.5 * @iqr_unit)')



sns.scatterplot(filtered_unitprice['UnitPrice'], filtered_quantity['Quantity'])

plt.title('Quantity x UnitPrice', fontsize = 20);
q1_quan_custom = df['Quantity'].quantile(0.5)

q3_quan_custom = df['Quantity'].quantile(0.95)

iqr_quan_custom = q3_quan_custom - q1_quan_custom



q1_unit_custom = df['UnitPrice'].quantile(0.5)

q3_unit_custom = df['UnitPrice'].quantile(0.95)

iqr_unit_custom = q3_unit_custom - q1_unit_custom



dx = df[df['Quantity']>0] #hilangkan value negatif

dy = df[df['UnitPrice']>0] #hilangkan value negatif



filtered_quantity = dx.query('(@q1_quan_custom - 1.5 * @iqr_quan_custom) <= Quantity <= (@q3_quan_custom + 1.5 * @iqr_quan_custom)')

filtered_unitprice = dy.query('(@q1_unit_custom - 1.5 * @iqr_unit_custom) <= UnitPrice <= (@q3_unit_custom + 1.5 * @iqr_unit_custom)')



sns.scatterplot(filtered_unitprice['UnitPrice'], filtered_quantity['Quantity'])

plt.title('Quantity x UnitPrice', fontsize = 20);
sns.distplot(df['Quantity'])

plt.title('Distribusi Quantity', fontsize = 20)

plt.xlabel('Quantity')

plt.ylabel('count');
sns.distplot(df['UnitPrice'])

plt.title('Distribusi Unit price', fontsize = 20)

plt.xlabel('Unit Price')

plt.ylabel('count');
x = df['Country'].value_counts().head(5)

sns.barplot(x = x.values, y = x.index, )

plt.title('5 negara terbesar', fontsize = 20)

plt.xlabel('Count')

plt.ylabel('Nama Negara');
x = df['Country'].nunique()

print("Terdapat total {} negara".format(x))



country = pd.DataFrame(df['Country'].value_counts()).reset_index()

country.columns = ['Negara', 'Jumlah Transaksi']

country
df = df[df['Country'] == 'United Kingdom']

df.head()
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

df.head()
df = df[~df['CustomerID'].isna()]

df.head()
df = df[df['Quantity']>0]

df = df[df['UnitPrice']>0]

df.head()
df['Revenue'] = df['Quantity'] * df['UnitPrice']

df.head()
orders = df.groupby(['InvoiceNo', 'InvoiceDate']).sum().reset_index()

orders.head()
rfm = summary_data_from_transaction_data(orders, 'CustomerID', 'InvoiceDate', monetary_value_col='Revenue').reset_index()

rfm
plt.hist(rfm['frequency'])

plt.title('Frequency')

plt.ylabel('Jumlah Customer' )

plt.xlabel('Frequency');
rfm = rfm[rfm['frequency']>0]

rfm.head()
plt.hist(rfm['frequency'])

plt.title('Frequency')

plt.ylabel('Jumlah Customer', )

plt.xlabel('Frequency');
plt.hist(rfm['monetary_value'])

plt.title('Monetary Value')

plt.ylabel('Jumlah Customer', )

plt.xlabel('Monetary Value');
rfm = rfm[rfm['monetary_value']<2000]

rfm.head()
plt.hist(rfm['monetary_value'])

plt.title('Monetary Value')

plt.ylabel('Jumlah Customer', )

plt.xlabel('Monetary Value');
quartiles = rfm.quantile(q=[0.25, 0.5, 0.75])

quartiles
def recency_score (data):

    if data <= 60:

        return 1

    elif data <= 128:

        return 2

    elif data <= 221:

        return 3

    else:

        return 4



def frequency_score (data):

    if data <= 1:

        return 1

    elif data <= 1:

        return 2

    elif data <= 2:

        return 3

    else:

        return 4



def monetary_value_score (data):

    if data <= 142.935:

        return 1

    elif data <= 292.555:

        return 2

    elif data <= 412.435:

        return 3

    else:

        return 4



rfm['R'] = rfm['recency'].apply(recency_score )

rfm['F'] = rfm['frequency'].apply(frequency_score)

rfm['M'] = rfm['monetary_value'].apply(monetary_value_score)

rfm.head()
rfm['RFM_score'] = rfm[['R', 'F', 'M']].sum(axis=1)

rfm.head()
rfm['label'] = 'Bronze' 

rfm.loc[rfm['RFM_score'] > 4, 'label'] = 'Silver' 

rfm.loc[rfm['RFM_score'] > 6, 'label'] = 'Gold'

rfm.loc[rfm['RFM_score'] > 8, 'label'] = 'Platinum'

rfm.loc[rfm['RFM_score'] > 10, 'label'] = 'Diamond'



rfm.head()
barplot = dict(rfm['label'].value_counts())

bar_names = list(barplot.keys())

bar_values = list(barplot.values())

plt.bar(bar_names,bar_values)

print(pd.DataFrame(barplot, index=[' ']))
df2 = pd.read_csv("/kaggle/input/onlineretail/OnlineRetail.csv",error_bad_lines=False, encoding='unicode_escape')

df2.head()
df2['InvoiceDate'] = pd.to_datetime(df2['InvoiceDate'])

df2.head()
uk = df2[df2['Country'] == 'United Kingdom']

uk.head()
main_df = pd.DataFrame(df2['CustomerID'].unique())

main_df.columns = ['CustomerID']

main_df.head()
latest_purchase = uk.groupby('CustomerID').InvoiceDate.max().reset_index()

latest_purchase.columns = ['CustomerID','LatestPurchaseDate']

latest_purchase.head()
latest_purchase['Recency'] = (latest_purchase['LatestPurchaseDate'].max() - latest_purchase['LatestPurchaseDate']).dt.days

latest_purchase.head()
main_df = pd.merge(main_df, latest_purchase[['CustomerID','Recency']], on='CustomerID')

main_df.head()
sns.distplot(main_df['Recency'], kde=False, bins=50)

plt.title('Distribusi Value Recency', fontsize = 20)

plt.xlabel('Recency')

plt.ylabel('count');
from sklearn.cluster import KMeans
score = []

for k in range(1, 15):

    kmeans = KMeans(n_clusters=k)

    member = kmeans.fit_predict(np.array(main_df['Recency']).reshape(-1, 1))

    score.append(kmeans.inertia_)

    

plt.figure(figsize=(10, 5))

plt.plot(range(1, 15), score)

plt.ylabel("Inertia")

plt.xlabel("n_clusters");
kmeans = KMeans(n_clusters=4)

kmeans.fit(main_df[['Recency']])

main_df['RecencyCluster'] = kmeans.predict(main_df[['Recency']])

main_df.head()
main_df.groupby('RecencyCluster')['Recency'].describe()
def order_cluster(cluster_field_name, target_field_name,df,ascending):

    new_cluster_field_name = 'new_' + cluster_field_name

    df_new = df.groupby(cluster_field_name)[target_field_name].mean().reset_index()

    df_new = df_new.sort_values(by=target_field_name,ascending=ascending).reset_index(drop=True)

    df_new['index'] = df_new.index

    df_final = pd.merge(df,df_new[[cluster_field_name,'index']], on=cluster_field_name)

    df_final = df_final.drop([cluster_field_name],axis=1)

    df_final = df_final.rename(columns={"index":cluster_field_name})

    return df_final
main_df = order_cluster('RecencyCluster', 'Recency',main_df,False)

main_df.head()
main_df.groupby('RecencyCluster')['Recency'].describe()
frequency = uk.groupby('CustomerID').InvoiceDate.count().reset_index()

frequency.columns = ['CustomerID','Frequency']

frequency.head()
main_df = pd.merge(main_df, frequency, on='CustomerID')

main_df.head()
main_df.Frequency.describe()
sns.distplot(main_df['Frequency'], kde=False, bins=50)

plt.title('Distribusi Value Frequency', fontsize = 20)

plt.xlabel('Frequency')

plt.ylabel('count');
score = []

for k in range(1, 15):

    kmeans = KMeans(n_clusters=k)

    member = kmeans.fit_predict(np.array(main_df['Frequency']).reshape(-1, 1))

    score.append(kmeans.inertia_)

    

plt.figure(figsize=(10, 5))

plt.plot(range(1, 15), score)

plt.ylabel("Inertia")

plt.xlabel("n_clusters");
kmeans = KMeans(n_clusters=4)

kmeans.fit(main_df[['Frequency']])

main_df['FrequencyCluster'] = kmeans.predict(main_df[['Frequency']])

main_df.head()
main_df = order_cluster('FrequencyCluster', 'Frequency',main_df,True)

main_df.head()
main_df.groupby('FrequencyCluster')['Frequency'].describe()
uk['Revenue'] = uk['UnitPrice'] * uk['Quantity']

revenue = uk.groupby('CustomerID').Revenue.sum().reset_index()

revenue.head()
main_df = pd.merge(main_df, revenue, on='CustomerID')

main_df.head()
main_df['Revenue'].describe()
sns.distplot(main_df['Revenue'], kde=False, bins=50)

plt.title('Distribusi Revenue', fontsize = 20)

plt.xlabel('Revenue')

plt.ylabel('count');
score = []

for k in range(1, 15):

    kmeans = KMeans(n_clusters=k)

    member = kmeans.fit_predict(np.array(main_df['Revenue']).reshape(-1, 1))

    score.append(kmeans.inertia_)

    

plt.figure(figsize=(10, 5))

plt.plot(range(1, 15), score)

plt.ylabel("Inertia")

plt.xlabel("n_clusters");
kmeans = KMeans(n_clusters=4)

kmeans.fit(main_df[['Revenue']])

main_df['RevenueCluster'] = kmeans.predict(main_df[['Revenue']])

main_df.head()
main_df = order_cluster('RevenueCluster', 'Revenue',main_df,True)

main_df.head()
main_df.groupby('RevenueCluster')['Revenue'].describe()
main_df['RFM_score'] = main_df['RecencyCluster'] + main_df['FrequencyCluster'] + main_df['RevenueCluster']

main_df.head()
main_df['RFM_score'].unique()
main_df.groupby('RFM_score')['Recency','Frequency','Revenue'].mean()

main_df.head()
main_df['label'] = 'Bronze' 

main_df.loc[main_df['RFM_score'] > 1, 'label'] = 'Silver' 

main_df.loc[main_df['RFM_score'] > 2, 'label'] = 'Gold'

main_df.loc[main_df['RFM_score'] > 3, 'label'] = 'Platinum'

main_df.loc[main_df['RFM_score'] > 5, 'label'] = 'Diamond'



main_df.head()
barplot = dict(main_df['label'].value_counts())

bar_names = list(barplot.keys())

bar_values = list(barplot.values())

plt.bar(bar_names,bar_values)

print(pd.DataFrame(barplot, index=[' ']))