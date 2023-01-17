#import library

import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

import warnings

import missingno as msno

import datetime as dt

from PIL import Image

from IPython.display import Image
# set to options



# to display all columns and rows:

pd.set_option('display.max_columns', None); 

pd.set_option('display.max_rows', None);



# number of digits after comma

pd.set_option('display.float_format', lambda x: '%.2f' % x)



# for warnings

warnings.filterwarnings("ignore")

warnings.simplefilter(action='ignore', category=FutureWarning)
# reading dataset

data = pd.read_excel("../input/uci-online-retail-ii-data-set/online_retail_II.xlsx", sheet_name = "Year 2010-2011")
df = data.copy()

df.head()
df.info()
# ıs there any null values

df.isnull().values.any()
df.isnull().head()
# count of null values by variables

df.isnull().sum()
msno.bar(df)

plt.show()
msno.matrix(df)

plt.show()
df.head()
# number of unique items 

df["Description"].nunique()
# number of items by products

df["Description"].value_counts().head()
# number of orders by product (minus : refund items)

df.groupby("Description").agg({"Quantity" : "sum"}).head()
# most buying things first 5 

df.groupby("Description").agg({"Quantity" : "sum"}).sort_values("Quantity", ascending = False).head()
df.describe().T
# change column name 

df.rename(columns = {"Customer ID" : "CustomerID"}, inplace = True)
# new creating variable

df["TotalPrice"] = df["Quantity"] * df["Price"]
df.head()
# refund items 

df[df["Invoice"].str.contains("C", na = False)].head()
df_refund = df[df["Invoice"].str.contains("C", na = False)]
df_new = df[~df["Invoice"].str.contains("C", na = False)]
df_new.head()
#the amount of money paid according to the invoice first 5 (sorted)

df_new.groupby("Invoice").agg({"TotalPrice" : "sum"}).sort_values("TotalPrice", ascending = False).head()
#order amount by country(sorted)

df_new.groupby("Country").agg({"Quantity" : "count"}).sort_values("Quantity", ascending=False).head()
# amount of money earned by country

df_new.groupby("Country").agg({"TotalPrice" : "sum"}).sort_values("TotalPrice", ascending=False).head()
df_refund.head()
# count of null values by variables without refund items

df_new.isnull().sum()
# ignore null values

df_new.dropna(inplace= True)
df_new.isnull().sum()
#observations - variables

df_new.shape
df_new.describe([0.01,0.05,0.10,0.25,0.50,0.75,0.90,0.95, 0.99]).T
df1 = df_new
df1["InvoiceDate"].min()
df1["InvoiceDate"].max()
# determine as today

today_date = dt.datetime(2010, 12, 9)
# last invoice date by customer

df1.groupby("CustomerID").agg({"InvoiceDate" : "max"}).head()
#convert float to int

df1["CustomerID"] = df1["CustomerID"].astype(int)
#broadcasting property on pandas

(today_date - df1.groupby("CustomerID").agg({"InvoiceDate" : "max"})).head()
temp_df = (today_date - df1.groupby("CustomerID").agg({"InvoiceDate" : "max"}))
temp_df.rename(columns = {"InvoiceDate" : "Recency"}, inplace = True)
temp_df.head()
recency_df = temp_df["Recency"].apply(lambda x: x.days)
recency_df.head()
# invoice by customer and invoice

df1.groupby(["CustomerID", "Invoice"]).agg({"Invoice" : "nunique"}).head()
freq_df = df1.groupby(["CustomerID"]).agg({"InvoiceDate" : "nunique"})
freq_df.rename(columns = {"InvoiceDate" : "Frequency"}, inplace = True)
freq_df.head()
df1.head()
df1.groupby("CustomerID").agg({"TotalPrice" : "sum"}).sort_values("TotalPrice", ascending=False).head()
monetary_df = df1.groupby("CustomerID").agg({"TotalPrice" : "sum"})
monetary_df.rename(columns = {"TotalPrice" : "Monetary"}, inplace = True)
monetary_df.head()
rfm = pd.concat([recency_df, freq_df, monetary_df], axis = 1)
rfm.head()
#set values to rfm score

rfm["RecencyScore"] = pd.qcut(rfm["Recency"], 5, labels = [5,4,3,2,1])
rfm["FrequencyScore"] = pd.qcut(rfm["Frequency"].rank(method = "first"), 5, labels = [1,2,3,4,5])
rfm["MonetaryScore"] = pd.qcut(rfm["Monetary"], 5, labels = [1,2,3,4,5])
rfm.head()
# concanate rfm score

rfm["RFM_SCORE"] = (rfm['RecencyScore'].astype(str) + 

 rfm['FrequencyScore'].astype(str) + 

 rfm['MonetaryScore'].astype(str))
rfm.head()
#champion group

rfm[rfm["RFM_SCORE"] == "555"].head()
rfm.describe().T
#hibernating group

rfm[rfm["RFM_SCORE"] == "111"].head()
# map of rfm using regex expression

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
rfm["Segment"] = rfm["RecencyScore"].astype(str) + rfm["FrequencyScore"].astype(str)
rfm.head()
rfm['Segment'] = rfm['Segment'].replace(seg_map, regex=True)
rfm.head()
rfm[["Segment","Recency","Frequency", "Monetary"]].groupby("Segment").agg(["mean","median","std","count"])

#rfm.groupby(["Segment","Recency","Frequency","Monetary"]).agg(["mean","count"])