import pandas as pd

import numpy as np

import matplotlib.pyplot as plt
df = pd.read_csv('../input/online-retail-ii-uci/online_retail_II.csv')
df = df[df["Customer ID"].notna()]
### Sub-Samples



df = df.sample(frac=1)[:20000]
df["InvoiceDate"].dtype
### Object to Datetime



df["InvoiceDate"] = df["InvoiceDate"].apply(lambda x: pd.to_datetime(x.split(" ")[0]))
### Today's date



import datetime



today = datetime.date.today()
today = pd.Timestamp(today)
### RFM values



rfm_table = df.groupby("Customer ID").agg({"InvoiceDate":lambda x: (today - x.max()).days,

                              "Customer ID":lambda x: len(x),

                              "Price":lambda x: sum(x)})

rfm_table
### Column name change

rfm_table.columns = ["Recency","Frequency","Monetary"]
### Clustring quantiles values

### +1 added for start from 1 (not 0)



quantiles = 4



rfm_table.sort_values('Recency',ascending=False)

rfm_table['Rec_Tile'] = pd.cut(rfm_table['Recency'],quantiles,labels=False) + 1



rfm_table.sort_values('Frequency',ascending=False)

rfm_table['Freq_Tile']=pd.cut(rfm_table['Frequency'],quantiles,labels=False) + 1



rfm_table.sort_values('Monetary',ascending=False)

rfm_table['Mone_Tile'] =pd.cut(rfm_table['Monetary'],quantiles,labels=False) + 1
rfm_table
### customers who buy most expensive products



rfm_table[rfm_table["Mone_Tile"] >= 3]
### customers who frequent buy most cheap products



rfm_table[(rfm_table["Freq_Tile"] >= 2) & (rfm_table["Mone_Tile"] <= 2)]