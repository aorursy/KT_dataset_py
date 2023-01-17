# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from mlxtend.frequent_patterns import apriori, association_rules

import matplotlib.pyplot as plt

import seaborn as sns

import datetime as dt



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_excel("../input/uci-online-retail-ii-data-set/online_retail_II.xlsx", sheet_name = "Year 2010-2011")
data.head()
# iadeleri sildik

"""

iadeler = []

for i,j in enumerate(df["Invoice"].values):

    if str(j).startswith("C"):

        iadeler.append(i)

    



df.drop(iadeler, inplace=True) 

"""

data=data[~data["Invoice"].astype(str).str.startswith("C")]
data.isna().sum()
ba = data.copy()
ba.dropna(subset = ['Description'], inplace=True) # nanları attık.
ba.isna().sum()
ba.head()
ba.shape
ba["Description"] = ba["Description"].astype(str).apply(lambda x: x.strip())
ba=ba[~ba["Description"].astype(str).str.startswith("wrong")]
ba.shape
# işimize yaramayan column'ları attık.

ba.drop(columns=['StockCode', "InvoiceDate", "Price", "Country", "Customer ID"], inplace=True)
ba['Description'].tail()
ba.head()
ba.groupby(['Invoice','Description'])['Description'].count()
# Invoice ve Description'a göre gruplayıp Quantityye göre toplayıp unstack yapıyor

branch_order = (ba

          .groupby(['Invoice', 'Description'])['Quantity'] 

          .sum().unstack().reset_index().fillna(0) 

          .set_index('Invoice'))
branch_order.head()
encoded = branch_order.applymap(lambda x: 1 if x != 0 else 0) # 
encoded.head()
freq_items = apriori(encoded, min_support=0.04, use_colnames=True, verbose=True)
freq_items
freq_items.sort_values('support', ascending=False)
association_rules(freq_items, metric = 'confidence', min_threshold=0.4).sort_values(['support','confidence'], ascending=[False,False])

rfm = data.copy()
rfm.head()
rfm.dropna(inplace=True)
rfm.drop(columns=['StockCode', 'Country'], inplace=True) # işimize yaramayan columnları attık.
rfm["Customer ID"] = rfm["Customer ID"].astype(int) # CustomerID'yi integer'a çevirdik. Çünkü çirkin duruyordu.
rfm.head()
# Fatura başına ortalama ne kadar kazanılmıştır? 

rfm['Total'] = rfm["Quantity"] * rfm['Price']
rfm.head()
# Fatura başı toplam kazanç

rfm.groupby('Invoice').agg({'Total':'sum'}) 
# Aykırı değerler var mı? Varsa kaç tane

for feature in ["Quantity","Price","Total"]:



    Q1 = rfm[feature].quantile(0.01)

    Q3 = rfm[feature].quantile(0.99)

    IQR = Q3-Q1

    upper = Q3 + 1.5*IQR

    lower = Q1 - 1.5*IQR



    if rfm[(rfm[feature] > upper) | (rfm[feature] < lower)].any(axis=None):

        print(feature,"yes")

        print(rfm[(rfm[feature] > upper) | (rfm[feature] < lower)].shape[0])

    else:

        print(feature, "no")
rfm['InvoiceDate'].min() # ilk tarih
rfm['InvoiceDate'].max() # son tarih
today_date = dt.datetime(2011, 12 ,9) #bugünün tarihi
# Müşteriler en son ne zaman alışveriş yaptı?

rfm.groupby("Customer ID").agg({"InvoiceDate":"max"}).head()
# Bugünden itibaren kaç gün önce alışveriş yapıldı?

(today_date - rfm.groupby("Customer ID").agg({"InvoiceDate":"max"})).head() 
temp_df = (today_date - rfm.groupby("Customer ID").agg({"InvoiceDate":"max"}))

temp_df.rename(columns={"InvoiceDate": "Recency"}, inplace = True)
temp_df.head()
recency_df = temp_df["Recency"].apply(lambda x: x.days) # Günleri aldık.
recency_df.head()
temp_df = rfm.groupby(["Customer ID","Invoice"]).agg({"Invoice":"count"})
temp_df.head()
# Her müşterinin kaç faturası var?

temp_df.groupby("Customer ID").agg({"Invoice":"count"}).head()
freq_df = temp_df.groupby("Customer ID").agg({"Invoice":"sum"})

freq_df.rename(columns={"Invoice": "Frequency"}, inplace = True)

freq_df.head()
monetary_df = rfm.groupby("Customer ID").agg({"Total":"sum"})
monetary_df.head()
monetary_df.rename(columns={"Total": "Monetary"}, inplace = True)
print(recency_df.shape,freq_df.shape,monetary_df.shape)
# rfm adında yeni bir DataFrame oluşturup recency, frequency ve monetary'yi birleştirdik.

rfm = pd.concat([recency_df, freq_df, monetary_df],  axis=1) 
rfm.head()
# Recency : En yakın tarihten en uzak tarihe göre 5'ten 1'e skorladık.

# Frequency : Sıklığa göre 1'den 5'e göre skorladık

# Monetary : Müşteriden kazanılan toplam paraya göre skorladık.

rfm["RecencyScore"] = pd.qcut(rfm['Recency'], 5, labels = [5, 4, 3, 2, 1])

rfm["FrequencyScore"] = pd.qcut(rfm['Frequency'], 5, labels = [1, 2, 3, 4, 5])

rfm["MonetaryScore"] = pd.qcut(rfm['Monetary'], 5, labels = [1, 2, 3, 4, 5])
rfm.head()
(rfm['RecencyScore'].astype(str) + 

 rfm['FrequencyScore'].astype(str) + 

 rfm['MonetaryScore'].astype(str)).head()
# Müşteri segmentlerini belirledik/tanımladık. 

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
# Regex'e göre Her müşteriyi segmentlere ayırdık.

rfm['Segment'] = rfm['RecencyScore'].astype(str) + rfm['FrequencyScore'].astype(str)

rfm['Segment'] = rfm['Segment'].replace(seg_map, regex=True)

rfm.head()
rfm[["Segment", "Recency","Frequency","Monetary"]].groupby("Segment").agg(["mean","count"])
rfm[rfm["Segment"] == "Need Attention"].head()

need_att = pd.DataFrame()

need_att['Need Attention Customer ID'] = rfm[rfm['Segment'] == 'Need Attention'].index
need_att.to_csv('need_att.csv') # csv'ye çevirme