import pandas as pd

import numpy as np
df_2010_2011 = pd.read_excel("../input/online-retail-data-set-from-uci-ml-repo/Online Retail.xlsx")
df = df_2010_2011.copy()
df.head()#first five
df.nunique()#number of unique values in each column
df["Description"].value_counts()#shows that which product is found how many times in the dataframe.
df.groupby("Description").agg({"Quantity":"count"}).sort_values("Quantity",ascending=False)

#which product had been bought how many times
df["InvoiceNo"].nunique()#Unique values in Invoice column
df["TotalPrice"] = df["Quantity"]*df["UnitPrice"]

#creates a new column which is called TotalPrice. Multiplies Quantity and Price values for each row.
df.head()
df.groupby("InvoiceNo").agg({"TotalPrice":"sum"}).head()#Shows total price of each invoice.
df.sort_values("UnitPrice", ascending = False).head()#Sort price column. (higher to lower)
df["Country"].value_counts()#Which country ordered how many times?
df.groupby("Country").agg({"TotalPrice":"sum"}).sort_values("TotalPrice", ascending = False).head()

#Sorts each country's sum of TotalPrice.
df.isnull().sum()#Sums null values of each columns.
df.dropna(inplace = True)#drops null values from dataframe
df.shape# (rows, columns)
df.describe([0.01,0.05,0.10,0.25,0.50,0.75,0.90,0.95, 0.99]).T #descriptive statistics of dataframe
for feature in ["Quantity","UnitPrice","TotalPrice"]:



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

        

#To recognize if there is any outlier values. There are some outliers.
df.head()
df.info() # info of dataframe. There is not any null values. I have dropped them above.
df["InvoiceDate"].min()#oldest invoice date in the dataframe
df["InvoiceDate"].max()#newest invoice date in the dataframe
import datetime as dt # imports datetime library
today = dt.datetime(2011,12,9)

# I assigned today's date. Suppose that today's date is 2011-12-09
today
df.head()
df["CustomerID"] = df["CustomerID"].astype(int)# converts customer id float to int (17850.0 -> 17850)
df.head()
temp_df = (today - df.groupby("CustomerID").agg({"InvoiceDate":"max"}))

#For rfm analysis, I need Customer ID and InvoiceDate. So I make a subtraction (today's date - InvoiceDate)
temp_df
temp_df.rename(columns={"InvoiceDate":"Recency"},inplace=True)

#InvoiceDate column is Recency column anymore. Because it shows the recency value.
recency_df = temp_df["Recency"].apply(lambda x: x.days)

#takes the day value. I don't need hour and strings
recency_df.head()
temp_df = df.groupby(["CustomerID","InvoiceNo"]).agg({"InvoiceNo":"count"})

#finding frequency value of each customer. Lists invoices and counts them
temp_df.head()
freq_df = temp_df.groupby("CustomerID").agg({"InvoiceNo":"count"})#counts all invoices for each customer
freq_df
freq_df.rename(columns={"InvoiceNo":"Frequency"},inplace = True)#changes column name (Invoice -> Frequency)
freq_df
monetary_df = df.groupby("CustomerID").agg({"TotalPrice":"sum"})#shows each customer's total spendings
monetary_df.head()
monetary_df.rename(columns={"TotalPrice":"Monetary"},inplace=True)#changes column name (TotalPrice -> Monetary)
monetary_df.head()
print(recency_df.shape,freq_df.shape,monetary_df.shape)#shape of each dataframes
rfm = pd.concat([recency_df,freq_df,monetary_df],axis=1)#concatenate recency,frequency and monetary dataframes.
rfm
rfm["Recency Score"] = pd.qcut(rfm["Recency"],5,labels = [5,4,3,2,1])

#Divides all recency scores to 5 part. If recency(days) gets higher, recency score will be decreased.
rfm["Frequency Score"] = pd.qcut(rfm["Frequency"].rank(method="first"),5,labels = [1,2,3,4,5])

#Divides all frequency scores to 5 part. If frequency gets higher,frequency score will be increased.
rfm["Monetary Score"] = pd.qcut(rfm["Monetary"],5,labels = [1,2,3,4,5])

#Divides all monetary scores to 5 part. If monetary(spendings) gets higher, monetary score will be increased.
rfm.head()
rfm["Recency Score"].astype(str) + rfm["Frequency Score"].astype(str) + rfm["Monetary Score"].astype(str)

#sums these scores as a string. (e.g. 1 + 3 + 2 = 132 this is the RFM score)
rfm["RFM SCORE"] = rfm["Recency Score"].astype(str) + rfm["Frequency Score"].astype(str) + rfm["Monetary Score"].astype(str)

#adds RFM score as a column.
rfm.head()
rfm.describe()#descriptive stats of rfm
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

#prepare the rfm table with respect to recency-frequency grid.
rfm['Segment'] = rfm['Recency Score'].astype(str) + rfm['Frequency Score'].astype(str)

rfm['Segment'] = rfm['Segment'].replace(seg_map, regex=True)

#creates segment column and gives the customer's segment according to RFM score.



rfm.head()
rfm.groupby("Segment").agg(["count","mean","std","min","median","max"])

#count, mean, std, min, median, max values for segments
rfm[rfm["Segment"] == "Need Attention"]

#shows Need Attention segment's customers