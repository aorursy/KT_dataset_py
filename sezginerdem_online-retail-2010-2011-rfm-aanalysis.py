import pandas as pd
import numpy as np
import seaborn as sns
pd.set_option('display.max_columns', None); pd.set_option('display.max_rows', None);

pd.set_option('display.float_format', lambda x: '%.0f' % x)
import matplotlib.pyplot as plt
df_2010_2011 = pd.read_excel("../input/online-retail-ii-data-set-from-ml-repository/online_retail_II.xlsx", sheet_name = "Year 2010-2011")
df = df_2010_2011.copy()
df.head()
# Unique products
df["Description"].nunique()
# Each products counts are..
df["Description"].value_counts().head()
# Best-seller
df.groupby("Description").agg({"Quantity":"sum"}).head()
df.groupby("Description").agg({"Quantity":"sum"}).sort_values("Quantity", ascending = False).head()
# Unique invoice
df["Invoice"].nunique()
df["TotalPrice"] = df["Quantity"]*df["Price"]
df.head()
# The top invoices for price
df.groupby("Invoice").agg({"TotalPrice":"sum"}).head()
# The most expensive product is "POSTAGE"
df.sort_values("Price", ascending = False).head()
df["Country"].value_counts().head()
df.isnull().sum()
df.dropna(inplace = True)
df.shape
df.describe([0.01,0.05,0.10,0.25,0.50,0.75,0.90,0.95, 0.99]).T
for feature in ["Quantity","Price","TotalPrice"]:

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
df["InvoiceDate"].min()
df["InvoiceDate"].max()
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
monetary_df = df.groupby("Customer ID").agg({"TotalPrice":"sum"})
monetary_df.head()
monetary_df.rename(columns={"TotalPrice": "Monetary"}, inplace = True)
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
rfm.describe().T
rfm[rfm["RFM_SCORE"] == "555"].head()
rfm[rfm["RFM_SCORE"] == "111"].head()
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
rfm[rfm["Segment"] == "New Customers"].index
new_df = pd.DataFrame()
new_df["NewCustomerID"] = rfm[rfm["Segment"] == "New Customers"].index
new_df.head()
new_df.to_csv("new_customers.csv")