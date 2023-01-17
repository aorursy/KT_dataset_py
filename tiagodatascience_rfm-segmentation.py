# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd
import numpy as np
import seaborn as sns
from scipy import stats
from datetime import date,timedelta

import plotly.express as px

from yellowbrick.cluster import KElbowVisualizer,SilhouetteVisualizer,InterclusterDistance
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
df_orders = pd.read_csv('/kaggle/input/brazilian-ecommerce/olist_orders_dataset.csv')
df_payments = pd.read_csv('/kaggle/input/brazilian-ecommerce/olist_order_payments_dataset.csv')
#columns and rows
df_orders.shape
#list columns
df_orders.columns
#data types
df_orders.dtypes
#sample for dataset
df_orders.head(5)
#convert ['order_purchase_timestamp','order_approved_at', 'order_delivered_carrier_date','order_delivered_customer_date', 'order_estimated_delivery_date'] to datetime
date_columns = ['order_purchase_timestamp','order_approved_at', 'order_delivered_carrier_date','order_delivered_customer_date', 'order_estimated_delivery_date']
for col in date_columns:
    df_orders[col] = pd.to_datetime(df_orders[col])
df_orders.dtypes
#view date range
df_orders[date_columns].describe()
df_orders.isna().sum()
df_orders['order_status'].value_counts()
#filter orders
df_delivered = df_orders.query('order_status == "delivered"')
#check nan values in new dataframe
df_delivered.isna().sum()
#list columns
df_payments.columns
#show sample
df_payments.head(5)
#size and columns
df_payments.shape
#check if exists duplicated orders
df_payments.duplicated().value_counts()
sns.distplot(df_payments['payment_value'],bins=20)
#check statistical data
df_payments.describe()
#check outliers
sns.boxplot(df_payments['payment_value'])
#remove outliers
z = np.abs(stats.zscore(df_payments['payment_value']))
df_payments_so = df_payments[(z < 3)]
df_outliers = df_payments[(z > 3)]
sns.boxplot(df_payments_so['payment_value'])
#outliers distribution
sns.boxplot(df_outliers['payment_value'])
#verify outliers count dropped
df_outliers.shape
#outliers stats
df_outliers.describe()
#convert order_id to index in both datasets
df_payments = df_payments.set_index('order_id')

df_orders = df_orders.set_index('order_id')
#Join datasets
order_payment = df_orders.join(df_payments)
#create RFM Data set
last_date = order_payment['order_delivered_carrier_date'].max() + timedelta(days=1)
#order_payment = order_payment.reset_index()
rfm = order_payment.groupby('customer_id').agg({
    'order_delivered_carrier_date': lambda x: (last_date - x.max()).days,
    'order_id': lambda x: len(x),
    'payment_value': 'sum'
})
rfm.dropna(inplace=True)
std = StandardScaler()
x_std = std.fit_transform(rfm)
model = KMeans()
visualizer = KElbowVisualizer(model, k=(4,12))

visualizer.fit(x_std)        # Fit the data to the visualizer
visualizer.show()  
model_k = KMeans(n_clusters=7)
kmeans = model_k.fit(x_std)
rfm['cluster'] = kmeans.labels_
rfm.columns = ['Recency','Frequency','MonetaryValue','cluster']
rfm.head()
px.scatter_3d(rfm,x='Recency',y='Frequency',z='MonetaryValue',color='cluster')
rfm.groupby('cluster').mean()