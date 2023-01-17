import pandas as pd

import numpy as np

import seaborn as sns

pd.set_option("display.max_columns",None);

pd.set_option("display.max_rows",None);

retail=pd.read_excel("../input/online-retail-ii-dataset/online_retail_II.xlsx", sheet_name="Year 2010-2011")

df=retail.copy()
df.head(5)
df.info()
df.isna().sum()  
import missingno as msno

msno.bar(df);
df[df.isnull().any(axis=1)].shape
import missingno as msno

msno.heatmap(df);
df[df.isnull().any(axis=1)].shape
135080/541910
missingdf=df[(df["Description"].isnull()==True)  & (df["Customer ID"].isnull()==True)].head()

missingdf.head()
missingdf.groupby("Country").agg({"Price":np.mean})
df=df[df.notnull().all(axis=1)]

print(df.shape)
df.head()
df["Customer ID"]=df["Customer ID"].astype("int64")
df.head(1)
df[df["Invoice"].astype("str").str.get(0)=="C"].shape
df[df["Invoice"].astype("str").str.get(0)=="C"].head()
df=df[df["Invoice"].astype("str").str.get(0)!="C"]
df[df["Invoice"].astype(str).str.get(0)!="5"].head()
df[df["Quantity"]<0]
df.info()
df.Country.value_counts().head()
df[df.duplicated(["Description","Invoice"],keep=False)].head()
df=df.drop([125])

df[df.duplicated(["Description","Invoice"],keep=False)].head()
df.groupby("Country").agg({"Price":"sum"}).applymap('{:,.2f}'.format).sort_values(by="Price", ascending=True).head(10)
# Unique products
df["Description"].nunique()
# Each products counts are..
df.Description.value_counts().head() # counts of categorical values
# Best-seller
df.groupby("Description").agg({"Quantity":sum}).sort_values(by="Quantity", ascending=False).head()
# Unique invoice
df["Invoice"].nunique()
df["Total_price"]=df["Quantity"]*df["Price"]
df.head(5)
pd.set_option("display.float_format", lambda x: "%.2f" % x)
# The top invoices for price
df.groupby("Invoice").agg({"Total_price":sum}).head()
df.groupby("Invoice").agg({"Total_price":"sum"}).sort_values("Total_price", ascending=False).head(11)
# The highest invoice
df[df["Invoice"]==581483]
# The most expensive product is "POSTAGE"
df.sort_values("Price",ascending=False).head()
# Countries total prices
df.groupby("Country").agg({"Total_price":"sum"}).sort_values("Total_price",ascending=False).head()
print(df["InvoiceDate"].max())

print(df["InvoiceDate"].min())
import datetime as dt

today_date=dt.datetime(2011,12,10)

print(today_date)

df.groupby("Customer ID").agg({"InvoiceDate":max}).sort_values("InvoiceDate", ascending=False).head(3)
rec_df=today_date-df.groupby("Customer ID").agg({"InvoiceDate":max})

rec_df.head(3)
rec_df.rename(columns={"InvoiceDate": "Recency"}, inplace=True)

rec_df.head(3)
rec_df=rec_df["Recency"].apply(lambda x: x.days)

rec_df.head()
df.head(2)
df.groupby(["Customer ID","Invoice"]).agg({"Invoice":"count"}).head(5)
df[df["Customer ID"]==17841].head(10)
df.groupby("Customer ID").agg({"InvoiceDate":"nunique"}).head(10)
freq_df=df.groupby("Customer ID").agg({"InvoiceDate":"nunique"})

freq_df.head(10)

freq_df.rename(columns={"InvoiceDate": "Frequency"}, inplace=True)

freq_df.head(3)
monetary_df=df.groupby("Customer ID").agg({"Total_price":"sum"})

monetary_df.head(7)
monetary_df.rename(columns={"Total_price":"Monetary"}, inplace=True)

monetary_df.head()
rfm=pd.concat([rec_df,freq_df, monetary_df], axis=1)

rfm.head(7)
rfm["Recency_Score"]= pd.qcut(rfm["Recency"],5, labels=[5,4,3,2,1])
rfm["Frequency_Score"]= pd.qcut(rfm["Frequency"].rank(method="first"),5, labels=[1,2,3,4,5])

rfm["Monetary_Score"]=pd.qcut(rfm["Monetary"],5, labels=[1,2,3,4,5])
rfm.head(7)
rfm["RFM"]=rfm["Recency_Score"].astype(str)+rfm["Frequency_Score"].astype(str)+rfm["Monetary_Score"].astype(str)

rfm.head(10)
rfm.head(6)
seg_map={r'[1-2][1-2]': "Hibernating", r'[1-2][3-4]': "At Risk", r'[1-2]5': "Can't Lose", r'3[1-2]': "About to Sleep",

        r'33': "Need Attention", r'[3-4][4-5]': "Loyal Customers", r'41': "Promising", r'51': "New Customers",

        r'[4-5][2-3]': "Potential Loyalist", r'5[4-5]': "Champions"}
rfm["Segment"]=rfm["Recency_Score"].astype(str)+ rfm["Frequency_Score"].astype(str)

rfm["Segment"]=rfm["Segment"].replace(seg_map,regex=True)
rfm.head()
rfm[["Segment","Recency","Frequency","Monetary"]].groupby("Segment").agg(["min","max","mean","count"])
rfm[rfm["Segment"]=="Need Attention"].index
dff=pd.DataFrame()

dff["Need_Attention_ID"]=rfm[rfm["Segment"]=="Need Attention"].index

dff.head()
dff.to_csv("Need_Attention.csv")