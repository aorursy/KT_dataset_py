# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df_2010_2011 = pd.read_excel("../input/uci-online-retail-ii-data-set/online_retail_II.xlsx", sheet_name = "Year 2010-2011")

df = df_2010_2011.copy() # df adında değişkene kopya oluşturdum.
df.head() # df'in ilk 5 row'una bakalım.
df.tail() # df'in son 5 row'una da göz gezdirelim
# eşsiz ürün sayısı 
df['Description'].nunique()
# Hangi üründen kaçar tane var?
df['Description'].value_counts().head()
# Fatura başına ortalama ne kadar kazanılmıştır? 
df['Total'] = df["Quantity"] * df['Price']
df.head()
# Fatura başı ne kadar kazanıldı? (Yani her alışverişte ne kadar kazanıldı?)
df.groupby("Invoice").agg({'Total':"sum"})
# iadeleri sildik
iadeler = []
for i,j in enumerate(df["Invoice"].values):
    if str(j).startswith("C"):
        iadeler.append(i)
    

df.drop(iadeler, inplace=True) 
df.isnull().sum()
df.dropna(inplace = True)
df.shape
for feature in ["Quantity","Price","Total"]:

    Q1 = df[feature].quantile(0.01)
    Q3 = df[feature].quantile(0.99)
    IQR = Q3-Q1
    upper = Q3 + 1.5*IQR
    lower = Q1 - 1.5*IQR

    if df[(df[feature] > upper) | (df[feature] < lower)].any(axis=None):
        print(feature,"yes")
        print(df[(df[feature] > upper) | (df[feature] < lower)].shape[0])
    else:
        print(feature, "no")
df.head()
df.info()
df['InvoiceDate'].min() # ilk tarih
df['InvoiceDate'].max() # Son tarih
import datetime as dt
today_date = dt.datetime(2011,12,9)
today_date
df.groupby("Customer ID").agg({"InvoiceDate":"max"}).head()
df["Customer ID"] = df["Customer ID"].astype(int)
(today_date - df.groupby("Customer ID").agg({"InvoiceDate":"max"})).head()
temp_df = (today_date - df.groupby("Customer ID").agg({"InvoiceDate":"max"}))
temp_df.rename(columns={"InvoiceDate": "Recency"}, inplace = True)
temp_df.head()
recency_df = temp_df["Recency"].apply(lambda x: x.days)
recency_df.head()
temp_df = df.groupby(["Customer ID","Invoice"]).agg({"Invoice":"count"})
temp_df.head()
temp_df.groupby("Customer ID").agg({"Invoice":"sum"}).head()
freq_df = temp_df.groupby("Customer ID").agg({"Invoice":"sum"})
freq_df.rename(columns={"Invoice": "Frequency"}, inplace = True)
freq_df.head()
monetary_df = df.groupby("Customer ID").agg({"Total":"sum"})
monetary_df.head()
monetary_df.rename(columns={"Total": "Monetary"}, inplace = True)
print(recency_df.shape,freq_df.shape,monetary_df.shape)
rfm = pd.concat([recency_df, freq_df, monetary_df],  axis=1)
rfm.head()
rfm["RecencyScore"] = pd.qcut(rfm['Recency'], 5, labels = [5, 4, 3, 2, 1])
rfm["FrequencyScore"] = pd.qcut(rfm['Frequency'], 5, labels = [1, 2, 3, 4, 5])
rfm["MonetaryScore"] = pd.qcut(rfm['Monetary'], 5, labels = [1, 2, 3, 4, 5])
rfm.head()
(rfm['RecencyScore'].astype(str) + 
 rfm['FrequencyScore'].astype(str) + 
 rfm['MonetaryScore'].astype(str)).head()
rfm["RFM_SCORE"] = rfm['RecencyScore'].astype(str) + rfm['FrequencyScore'].astype(str) + rfm['MonetaryScore'].astype(str)
rfm.head()
seg_map = {
    r'[1-2][1-2]': 'Hibernating',
    r'[1-2][3-4]': 'At Risk',
    r'[1-2]5': 'Can\'t Loose',
    r'3[1-2]': 'About to Sleep',
    r'33': 'Need Attention',
    r'[3-4][4-5]': 'Loyal Customers',
    r'41': 'Promising',
    r'51': 'New Customers',
    r'[4-5][2-3]': 'Potential Loyalists',
    r'5[4-5]': 'Champions'
}
rfm['Segment'] = rfm['RecencyScore'].astype(str) + rfm['FrequencyScore'].astype(str)
rfm['Segment'] = rfm['Segment'].replace(seg_map, regex=True)
rfm.head()
rfm[["Segment", "Recency","Frequency","Monetary"]].groupby("Segment").agg(["mean","count"])
rfm[rfm["Segment"] == "Need Attention"].head()
need_att = pd.DataFrame()
need_att['Need Attention Customer ID'] = rfm[rfm['Segment'] == 'Need Attention'].index
need_att.head()
need_att.to_csv('need_att.csv')
