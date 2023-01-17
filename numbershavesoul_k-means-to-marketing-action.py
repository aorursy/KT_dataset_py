# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# load the packages
import seaborn as sns; sns.set(rc={'figure.figsize':(16,9)}) #visualization tool
import datetime as dt
import matplotlib.pyplot as plt
customer = pd.read_csv('../input/olist_customers_dataset.csv')
customer.head()
orders = pd.read_csv('../input/olist_orders_dataset.csv')

orders['order_approved_at2'] = pd.to_datetime(orders['order_approved_at'])
orders.head()
pagamentos = pd.read_csv('../input/olist_order_payments_dataset.csv')
pagamentos.head()
df = pd.merge(customer, orders, on ='customer_id')
df = pd.merge(df,pagamentos, on = 'order_id')
df.head()
df.info()
df = df[pd.notnull(df['order_approved_at'])]
df = df[pd.notnull(df['order_delivered_carrier_date'])]
df = df[pd.notnull(df['order_delivered_customer_date'])]
df = df[pd.notnull(df['order_approved_at2'])]
df.isnull().sum(axis=0)
# time since last purchase 
# the data from october 2016 to august 2018, so let's use september 2018 as cutoff date.
Data_Corte = dt.datetime(2018,9,13)

df['order_approved_at3'] = df['order_approved_at2']
# Criando Recencia, Frequencia e Valor Total gasto - Visão cliente (cada linha do df é um cliente)
df1 = df.groupby('customer_unique_id').agg({'order_approved_at2': lambda x: (Data_Corte - x.max()).days, 'order_id': lambda x: len(x), 'payment_value': lambda x: x.sum(),'order_approved_at3': lambda x: (Data_Corte - x.min()).days})



df1['order_approved_at2'] = df1['order_approved_at2'].astype(int)
df1['order_approved_at3'] = df1['order_approved_at3'].astype(int)

df1.rename(columns={'order_approved_at2': 'Recency', 
                    'order_id': 'Frequency', 
                    'payment_value': 'Monetary_Value',
                    'order_approved_at3': 'Client_Since'}, inplace=True)

df1.head()
df1.info()
df1.describe()
from sklearn.cluster import KMeans
from sklearn import preprocessing
wcss = []
 
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'random')
    kmeans.fit(df1)
    #print (i,kmeans.inertia_)
    wcss.append(kmeans.inertia_)  
plt.plot(range(1, 11), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('WSS') #within cluster sum of squares
plt.show()
scaler = preprocessing.StandardScaler()
scaled_df = scaler.fit_transform(df1) # variables normalized

kmeans  = KMeans(n_clusters = 5, random_state= 84635)
kmeans.fit(scaled_df)
kmeans.predict(scaled_df)

df1['prediction']=kmeans.predict(scaled_df)

freq =df1.groupby(['prediction'])

freq = freq ['prediction'].agg('count')
freq
y_kmeans = kmeans.predict(scaled_df)

plt.scatter(scaled_df[:, 0], scaled_df[:, 1], c=y_kmeans, s=100, cmap='viridis')

centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=500, alpha=0.5);
medidas=['count','min', 'mean', 'median', 'max','std']

qtd_preco = df1.groupby(['prediction'])

resumo = qtd_preco['Recency','Frequency','Monetary_Value', 'Client_Since'].agg(medidas)
resumo