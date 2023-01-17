import pandas as pd
import numpy as np
df_2010_2011 = pd.read_excel("../input/online-retail-ii/online_retail_II.xlsx", sheet_name = "Year 2010-2011")
df = df_2010_2011.copy()
df.head()
# Top five data
df.info()
df.nunique()
# Number of unique values in each column
df["TotalPrice"] = df["Quantity"] * df["Price"]
df.head()
df.isnull().values.any()
# Query of empty columns
df.isnull().sum()
# Total number of empty rows for each column
df = df[~df["Invoice"].str.contains("C", na = False)]
# Creates a dataframe with no return invoices
df.dropna(inplace = True)
# Null values are destroyed
df.isnull().sum()
df["Customer ID"] = df["Customer ID"].astype(int)
# Change the data type from float to integer
df.info()
df["InvoiceDate"].max()
# Invoice date of the last purchase
import datetime as dt
today_date = dt.datetime(2011, 12, 9)
# The latest purchase date has been assigned as today's date
today_date
temp_df = (today_date - df.groupby("Customer ID").agg({"InvoiceDate":"max"}))
# The information about how many days have passed since the last shopping date of each customer
temp_df.head()
temp_df.rename(columns = {"InvoiceDate":"Recency"}, inplace = True)
# The column name is changed because the value we found is actually the Recency value.
temp_df.head()
recency_df = temp_df["Recency"].apply(lambda x: x.days)
# Only the number of days is taken. We do not need time or string.
recency_df.head()
df.groupby(["Customer ID","Invoice"]).agg({"Invoice":"nunique"}).head()
# The invoice number made by each customer gives us the value of Frequency.
freq_df = df.groupby("Customer ID").agg({"Invoice":"nunique"})
# Frequency value is assigned as freq_df.
freq_df.head()
freq_df.rename(columns={"Invoice": "Frequency"}, inplace=True)
# The column name is changed.
freq_df.head()
monetary_df = df.groupby("Customer ID").agg({"TotalPrice":"sum"})
# It shows the total spending of each customer.
monetary_df.head()
monetary_df.rename(columns={"TotalPrice":"Monetary"}, inplace=True)
monetary_df.head()
print(recency_df.shape,freq_df.shape,monetary_df.shape)
# It is checked whether the number of rows is equal.
rfm = pd.concat([recency_df, freq_df, monetary_df],  axis=1)
# All the values found are concatenated in a dataframe. axis = 1 provides column-based side-by-side.
rfm.head()
rfm["RecencyScore"] = pd.qcut(rfm["Recency"], 5, labels = [5, 4 , 3, 2, 1])
rfm["FrequencyScore"]= pd.qcut(rfm["Frequency"].rank(method="first"),5, labels=[1,2,3,4,5])
rfm["MonetaryScore"] = pd.qcut(rfm['Monetary'], 5, labels = [1, 2, 3, 4, 5])
# The above three codes have been written to assign scores for values. Each value was given from 1 to 5.
# Since the small Recency value is good for us, the smallest value is assigned the largest. Therefore, a value from 5 to 1 is assigned.
rfm.head()
(rfm['RecencyScore'].astype(str) + 
 rfm['FrequencyScore'].astype(str) + 
 rfm['MonetaryScore'].astype(str)).head()
# Sums these scores as a string. (e.g. 1 + 3 + 2 = 132)
rfm["RFM_Score"] = (rfm['RecencyScore'].astype(str) + 
                    rfm['FrequencyScore'].astype(str) + 
                    rfm['MonetaryScore'].astype(str))
# Adds RFM score as column.
rfm.head()
rfm.describe().T
# Descriptive statistics of RFM
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
# A regex table has been created according to the Recency-Frequency values to divide into segments.
# It can also be done with if, but the regular expression (regex) is preferred because doing it with if takes much of time.
rfm['Segment'] = rfm['RecencyScore'].astype(str) + rfm['FrequencyScore'].astype(str)
rfm['Segment'] = rfm['Segment'].replace(seg_map, regex=True)
# Creates the segment column and returns the customer's segment based on the RFM score.
rfm.head()
rfm[["Segment","Recency","Frequency", "Monetary"]].groupby("Segment").agg(["mean","median","count"])
# Display mean, median, count values for segments
rfm[rfm["Segment"] == "At Risk"]
# As an example, it returns data from the "At Risk" segment.