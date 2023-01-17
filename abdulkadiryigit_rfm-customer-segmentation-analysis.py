import pandas  as pd 

import numpy   as np

import seaborn as sns

import matplotlib.pyplot as plt

import datetime as dt
#Read the data that I uploaded to Kaggle.

df_2010_2011 = pd.read_excel("/kaggle/input/online_retail_II.xlsx" ,sheet_name = "Year 2010-2011")
df=df_2010_2011.copy()
df.head()
#Delete NA values on the dataset

df.dropna(inplace = True)
df.shape
#Find maximum date of Invoices

df["InvoiceDate"].max()
# Create a datetime for last day of Invoices  

today_date = dt.datetime(2011,12,9)

today_date
# Analyse Customers what was the last day of their Invoices

df.groupby("Customer ID").agg({"InvoiceDate":"max"}).head()
df["Customer ID"]=df["Customer ID"].astype(int)
df_last_sales_date=(today_date-df.groupby("Customer ID").agg({"InvoiceDate":"max"}))
df_last_sales_date.rename(columns = {"InvoiceDate" : "Recency"}, inplace=True)
df_recency=df_last_sales_date["Recency"].apply(lambda x : x.days)
df_recency.head()
df_freq=df.groupby(["Customer ID","Invoice"]).agg({"Invoice":"count"})

df_freq.head()
df_freq=df_freq.groupby("Customer ID").agg({"Invoice":"sum"})
df_freq.rename(columns = ({"Invoice":"Frequency"}),inplace =True)

df_freq.head()
df.head()
df["TotalPrice"] = df["Quantity"]*df["Price"]

df
df_monetary=df.groupby("Customer ID").agg({"TotalPrice":"sum"})
df_monetary.rename(columns ={"TotalPrice":"Monetary"},inplace=True)

df_monetary.head()
rfm = pd.concat([df_recency, df_freq, df_monetary],  axis=1)
rfm.head()
rfm["RecencyScore"]   = pd.qcut(rfm['Recency'], 5,   labels = [5, 4, 3, 2, 1])

rfm["FrequencyScore"] = pd.qcut(rfm["Frequency"], 5, labels = [5, 4, 3, 2, 1]) 

rfm["MonetaryScore"] = pd.qcut(rfm['Monetary'], 5, labels = [1, 2, 3, 4, 5])

rfm.head()
rfm["RFM_SCORE"]=(rfm['RecencyScore'].astype(str) + rfm['FrequencyScore'].astype(str) + rfm['MonetaryScore'].astype(str))
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
rfm[["Segment","Recency","Frequency","Monetary"]].groupby("Segment").agg(["mean","count"])
rfm[rfm["Segment"] == "Need Attention"].head()
new_df = pd.DataFrame()

new_df["NewCustomerID"] = rfm[rfm["Segment"] == "New Customers"].index
asleep_df = pd.DataFrame()

asleep_df["AboutoSleepCustomerID"] = rfm[rfm["Segment"] == "About to Sleep"].index
rfm[rfm["Segment"] == "About to Sleep"].head()
rfm[rfm["Segment"] == "New Customers"].head()