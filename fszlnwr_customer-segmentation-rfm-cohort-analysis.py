import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns 
df = pd.read_csv('../input/data.csv', encoding = "ISO-8859-1")
df.head(10)
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate']) #ubah format InvoiceDate menjadi datetime
print("Informasi dari dataset :")

print("Jumlah Row \t\t:", df.shape[0]) #check jumlah total rows pada data

print("Jumlah Column \t\t:", df.shape[1]) #check jumlah total coloumns pada data

print("Date range from \t:", df.InvoiceDate.min(), " to ", df.InvoiceDate.max()) #check range waktu pada data

print("#Jumlah Transaksi \t:", df.InvoiceNo.nunique()) #check jumlah transaksi

print("#Unique Customer \t:", df.CustomerID.nunique()) #check jumlah unique customer

print("Range Quantity \t\t:", df.Quantity.min(), " to ", df.Quantity.max()) #check range Quantity pada data

print("Range UnitPrice \t:", df.UnitPrice.min(), " to ", df.UnitPrice.max()) #check range UnitPrice pada data
print(df.isnull().sum().sort_values(ascending=False))
df_new = df.dropna() ## remove null

df_new = df_new[df_new.Quantity > 0] ## remove negative value in Quantity column

df_new = df_new[df_new.UnitPrice > 0] ## remove negative value in UnitPrice column
print(df_new.isnull().sum().sort_values(ascending=False))
df_new['Revenue'] = df_new['Quantity'] * df_new['UnitPrice'] ## add Revenue (Qty * UnitPrice) column

df_new['CustomerID'] = df_new['CustomerID'].astype('int64') #change format CustomerID
import datetime as dt

NOW = dt.datetime(2011,12,10)
rfmTable = df_new.groupby('CustomerID').agg({'InvoiceDate': lambda x: (NOW - x.max()).days, 'InvoiceNo': lambda x: len(x), 'Revenue': lambda x: x.sum()})

rfmTable['InvoiceDate'] = rfmTable['InvoiceDate'].astype(int)

rfmTable.rename(columns={'InvoiceDate': 'recency', 

                         'InvoiceNo': 'frequency', 

                         'Revenue': 'monetary'}, inplace=True)
rfmTable.head()
quantiles = rfmTable.quantile(q=[0.25,0.5,0.75])

quantiles = quantiles.to_dict()

segmented_rfm = rfmTable
def RScore(x,p,d):

    if x <= d[p][0.25]:

        return 4

    elif x <= d[p][0.50]:

        return 3

    elif x <= d[p][0.75]: 

        return 2

    else:

        return 1

    

def FMScore(x,p,d):

    if x <= d[p][0.25]:

        return 1

    elif x <= d[p][0.50]:

        return 2

    elif x <= d[p][0.75]: 

        return 3

    else:

        return 4
segmented_rfm['r_quartile'] = segmented_rfm['recency'].apply(RScore, args=('recency',quantiles,))

segmented_rfm['f_quartile'] = segmented_rfm['frequency'].apply(FMScore, args=('frequency',quantiles,))

segmented_rfm['m_quartile'] = segmented_rfm['monetary'].apply(FMScore, args=('monetary',quantiles,))

segmented_rfm.head()
segmented_rfm['RFMScore'] = segmented_rfm.r_quartile.map(str)+segmented_rfm.f_quartile.map(str)+segmented_rfm.m_quartile.map(str)

segmented_rfm.head()
segmented_rfm[segmented_rfm['RFMScore']=='444'].sort_values('monetary', ascending=False).head()

top_customer = df_new[df_new['CustomerID'] == 14646]

top_customer.head(20)
df_new.head()
def get_month(x): return dt.datetime(x.year, x.month, 1)

df_new['InvoiceMonth'] = df_new['InvoiceDate'].apply(get_month)

grouping = df_new.groupby('CustomerID')['InvoiceMonth']

df_new['CohortMonth'] = grouping.transform('min')
df_new.head()
## function untuk extract integer value dari data

def get_date_int(df, column):

    year = df[column].dt.year

    month = df[column].dt.month

    day = df[column].dt.day

    return year, month, day
invoice_year, invoice_month, _ = get_date_int(df_new, 'InvoiceMonth')

cohort_year, cohort_month, _ = get_date_int(df_new, 'CohortMonth')
years_diff = invoice_year - cohort_year

months_diff = invoice_month - cohort_month
df_new['CohortIndex'] = years_diff * 12 + months_diff + 1
df_new.head()
## grouping customer berdasarkan masing masing cohort

grouping = df_new.groupby(['CohortMonth', 'CohortIndex'])

cohort_data = grouping['CustomerID'].apply(pd.Series.nunique)

cohort_data = cohort_data.reset_index()

cohort_counts = cohort_data.pivot(index='CohortMonth', columns='CohortIndex', values='CustomerID')
cohort_counts
cohort_sizes = cohort_counts.iloc[:,0]

retention = cohort_counts.divide(cohort_sizes, axis=0)

retention.round(2) * 100
plt.figure(figsize=(15, 8))

plt.title('Retention rates')

sns.heatmap(data = retention,

annot = True,

fmt = '.0%',

vmin = 0.0,

vmax = 0.5,

cmap = 'BuGn')

plt.show()