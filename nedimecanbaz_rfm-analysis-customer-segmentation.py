import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
### Warning settings
import warnings
warnings.simplefilter(action='ignore')
### Round numbers after comma to 2 numbers
import pandas as pd
pd.set_option('display.float_format', lambda x: '%.2f' % x)
pd.set_option('display.max_columns', None); 
pd.set_option('display.max_rows', None);
## Read Data
df_2010_2011 = pd.read_excel("../input/uci-online-retail-ii-data-set/online_retail_II.xlsx", sheet_name = "Year 2010-2011")
df = df_2010_2011.copy()
df.head()
### Let's identify the ones starting with C in invoice numbers and put them into a dataframe
df_Return = df[df["Invoice"].str.startswith("C", na = False)]
### Left real sales in the dataframe
df_Sales = df[~(df["Invoice"].str.startswith("C", na = False))]
#Analyse NA values on the dataset
df_Sales.isnull().sum()


## Observed that 1454 is "NaN" for Description "and 134697 is" NaN for "Customer ID"
## Delete NaN values on the dataset
df_Sales.dropna(subset=['Customer ID'], how='all', inplace=True)
df_Sales.isnull().sum()
# already dropped Nan Values from dataframe
### Observed some NaN in Return dataset, around 383 
df_Return.isnull().sum()
## Removed Nan from Return dataset
df_Return.dropna(subset=['Customer ID'], how='all', inplace=True)
df_Return.isnull().sum()

### Analyse data types how they maintained for each variable
df_Sales.info()

##Apply TIP transformation for "Customer_ID" to int type , get rid of comma
df_Sales["Customer ID"] = df_Sales["Customer ID"].astype(int)
df_Return["Customer ID"] = df_Return["Customer ID"].astype(int)
df_Sales.info()
## convert "Customer_ID" to categorical variable with using string method
df_Sales["Customer ID"] = df_Sales["Customer ID"].astype(str)
df_Return["Customer ID"] = df_Return["Customer ID"].astype(str)
## "Customer_ID" converted as categorical , shown 'object'
df_Sales.info()
## Released from comma
df_Sales.head()
## Size of new dataset
df_Sales.shape
#statistical information of numerical variables
df_Sales.describe().T
### what is the number of unique products
df_Sales["Description"].nunique()
##what is the number of unique products for Return
df_Return["Description"].nunique()
## how many of which products were sold?
df_Sales["StockCode"].value_counts().head()
## what is the most ordered product?
df_Sales.groupby("StockCode").agg({"Quantity":"sum"}).sort_values("Quantity",ascending=False).head()
##which are the most returned products?
df_Return.groupby("StockCode").agg({"Quantity":"sum"}).sort_values("Quantity",ascending=True).head()
## Total number of invoice
df_Sales["Invoice"].nunique()
## total price for each reco
df_Sales["TotalPrice"]=df_Sales["Quantity"]*df_Sales["Price"]
df_Sales.head()
## how many money was earned per invoice
df_Sales.groupby("Invoice").agg({"TotalPrice":"sum"}).sort_values("TotalPrice", ascending = False).head()

## which are the most expensive products
df_Sales.sort_values("Price",ascending = False).head()
## how many orders came from which country
df_Sales.groupby("Country").agg({"Quantity":"sum"}).sort_values("Quantity",ascending = False)

## Find maximum date of Invoices
Max_date= df_Sales["InvoiceDate"].max()
Max_date
## We should take the variable #Max_date as the datetime type. In this way, we will be able to perform the extraction between days
import datetime as dt
today_date = dt.datetime(2011,12,9,12,50,0)
today_date


## Let's take the last shopping dates of today's customers and assign the day values of the time between them to a new dataframe structure. These values are the customer's "Recency" values. Let's look at the top 5 observations
df_last_sales_date=(today_date-df_Sales.groupby("Customer ID").agg({"InvoiceDate":"max"})).rename(columns={"InvoiceDate":"Recency"})
df_recency=df_last_sales_date["Recency"].apply(lambda x : x.days)
df_recency.head()
df_Freq=df_Sales.groupby(["Customer ID","Invoice"]).agg({"Invoice":"count"})
df_Freq.head()
df_frequency = df_Freq.groupby("Customer ID").agg({"Invoice":"count"}).rename(columns = {"Invoice":"Frequency"})
df_frequency.head()
df_monetary = df_Sales.groupby("Customer ID").agg({"TotalPrice":"sum"}).rename(columns = {"TotalPrice":"Monetary"})
df_monetary.head()
##firstly, control that there is any mismatch issue between recency,frequency and monetary scores
print(df_recency.shape, df_frequency.shape, df_monetary.shape)
## apply Concatenate
df_rfm = pd.concat([df_recency,df_frequency,df_monetary], axis = 1)
df_rfm.head()
df_rfm["RecencyScore"]   = pd.qcut(df_rfm['Recency'], 5,   labels = [5, 4, 3, 2, 1])
df_rfm["FrequencyScore"] = pd.qcut(df_rfm["Frequency"].rank(method="first"),5, labels = [5, 4, 3, 2, 1]) 
df_rfm["MonetaryScore"] = pd.qcut(df_rfm['Monetary'], 5, labels = [1, 2, 3, 4, 5])
df_rfm.head()
#Add the RFM score on a customer basis and add RFM SCORE to our dataframe structure.
df_rfm["RFM_SCORE"] = df_rfm['RecencyScore'].astype(str) + df_rfm['FrequencyScore'].astype(str) + df_rfm['MonetaryScore'].astype(str)
df_rfm.head()
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
df_rfm['Segment'] = df_rfm['RecencyScore'].astype(str) + df_rfm['FrequencyScore'].astype(str)
df_rfm['Segment'] = df_rfm['Segment'].replace(seg_map, regex=True)
df_rfm.head()
need_attention_df = pd.DataFrame()
need_attention_df["NeedAttentionCustomerID"]= df_rfm[df_rfm["Segment"]=='Need Attention'].index
need_attention_df.head()
#Import to excel
need_attention_df.to_csv("Need_Attention.csv")
## I want to see the average and number values of each RFM group.
df_rfm[["Segment","Recency","Frequency","Monetary"]].groupby("Segment").agg(["mean","count"])
newcustomer_df = pd.DataFrame()
newcustomer_df["NewCustomerID"] = df_rfm[df_rfm["Segment"] == "New Customers"].index
df_rfm[df_rfm["Segment"] == "New Customers"].head()
Cantlose_df=pd.DataFrame()
Cantlose_df["cantlose"]=df_rfm[df_rfm["Segment"]=="Can't Loose"].index
df_rfm[df_rfm["Segment"] == "Can't Loose"].head()