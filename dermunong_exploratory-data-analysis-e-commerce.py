import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objs as go
import datetime as dt
from plotly.offline import init_notebook_mode,iplot
import warnings


warnings.simplefilter(action='ignore', category=FutureWarning)
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# Any results you write to the current directory are saved as output.# 1. Data Preparation
df = pd.read_csv('/kaggle/input/ecommerce-data/data.csv',encoding = "ISO-8859-1")
df.head()
df.count()
df.dtypes
df['InvoiceDate'] = df['InvoiceDate'].astype('datetime64[ns]')
df.describe()
df[df['Quantity']<0].head()
df[df['CustomerID']==17548]
cnt_order = df[df['Quantity']>0]['InvoiceNo'].nunique()
cnt_refund = df[df['Quantity']<0]['InvoiceNo'].nunique()


print("Total Orders : ",cnt_order)
print("Total Refund Order : ",cnt_refund)
print("%Refund : ",cnt_refund/(cnt_order)*100,"%")
df['RefundFlg'] = df['Quantity']<0
prod_order = df[df['Quantity']>0].groupby(['StockCode','Description']).InvoiceNo.nunique().sort_values(ascending = False).reset_index()
prod_order = prod_order.rename(columns = {'InvoiceNo' : 'TotalOrder'})
prod_order_refund = df[df['Quantity']<0].groupby(['StockCode','Description']).InvoiceNo.nunique().sort_values(ascending = False).reset_index()
prod_order_refund = prod_order_refund.rename(columns = {'InvoiceNo' : 'TotalRefundOrder'})

join_prod_order = prod_order.merge(prod_order_refund,left_on = ["StockCode","Description"],right_on = ["StockCode","Description"],how = 'left')
join_prod_order['%Refund'] = join_prod_order['TotalRefundOrder']/join_prod_order['TotalOrder']*100
join_prod_order = join_prod_order.sort_values(by = 'TotalRefundOrder',ascending = False)
join_prod_order.head()
join_prod_order = join_prod_order.sort_values(by = '%Refund',ascending = False)
join_prod_order[join_prod_order['TotalRefundOrder']>=10].head(10)
prod_only_order = join_prod_order[~join_prod_order['StockCode'].isin(['AMAZONFEE','S','BANK CHARGES','M'])]
TotalRefundProdOrder = prod_only_order['TotalRefundOrder'].sum()
TotalReProdOrder = prod_only_order['TotalOrder'].sum()
print("%Refund : ",TotalRefundProdOrder/TotalReProdOrder*100,"%")
df[df['UnitPrice']<0].head()
df['Net'] = df['Quantity']*df['UnitPrice']
sales_by_country = df.groupby(['Country']).Net.sum().sort_values(ascending = False).reset_index()
TotalSales = sales_by_country['Net'].sum()


sales_by_country['% of total sales'] = sales_by_country['Net']/TotalSales
#Top 5 countries by sales
sales_by_country.head()
df_wt_cust = df[(df['CustomerID'].notnull()) & (df['Quantity']>0)]

order_net = df_wt_cust.groupby(['InvoiceNo']).Net.sum()
aov = order_net.mean()
plt.hist(order_net,bins=1000)
plt.xlim(0,2000)
plt.xlabel("Order value")
plt.ylabel("Number of orders")
print("AOV : ",aov)
#Repeat Customers

#Exlude non-product rows

df_for_rpt_cust = df_wt_cust.copy()
df_for_rpt_cust = df_for_rpt_cust[~df_for_rpt_cust['StockCode'].isin(['AMAZONFEE','S','BANK CHARGES','M'])]

cust_wt_total_order = df_for_rpt_cust[df_for_rpt_cust['Net']>0].groupby(['CustomerID']).InvoiceNo.nunique().reset_index()
cust_wt_total_order = cust_wt_total_order.rename(columns = {'InvoiceNo' : 'TotalOrder'})

cust_wt_total_refund_order = df_for_rpt_cust[df_for_rpt_cust['Net']<0].groupby(['CustomerID']).InvoiceNo.nunique().reset_index()
cust_wt_total_refund_order = cust_wt_total_refund_order.rename(columns = {'InvoiceNo' : 'TotalRefundOrder'})

join_cust_wt_total_order = cust_wt_total_order.merge(cust_wt_total_refund_order,left_on = 'CustomerID',right_on='CustomerID',how = 'left')
# convert null to 0
join_cust_wt_total_order['TotalRefundOrder'] = np.where(join_cust_wt_total_order['TotalRefundOrder'].isnull(),0,join_cust_wt_total_order['TotalRefundOrder'])
join_cust_wt_total_order['TotalSuccessOrder'] = join_cust_wt_total_order['TotalOrder']-join_cust_wt_total_order['TotalRefundOrder']

join_cust_wt_total_order['RepeatFlg'] = join_cust_wt_total_order['TotalSuccessOrder']>=2

CntCustomer = join_cust_wt_total_order.CustomerID.nunique()
CntRepeatCustomer = join_cust_wt_total_order[join_cust_wt_total_order['RepeatFlg']==True].CustomerID.nunique()
PctRepeatCustomer = CntRepeatCustomer/CntCustomer

print("%Repeat Customer : ",PctRepeatCustomer*100,"%")

plt.hist(join_cust_wt_total_order['TotalSuccessOrder'],bins = 100)
plt.xlim(0,30)
plt.xlabel("Number of Orders")
plt.ylabel("Number of Customers")
plt.show()


df_wt_cust.groupby(['CustomerID','Description']).InvoiceNo.nunique().sort_values(ascending = False).reset_index().head(10)
sales_by_cust = df_wt_cust[df_wt_cust['InvoiceDate']>dt.date(2011,9,9)].groupby('CustomerID').Net.sum().sort_values(ascending=False).reset_index()
TotalSales = sales_by_cust.Net.sum()
sales_by_cust['%TotalSales'] = sales_by_cust['Net']/TotalSales


def cust_concentration(threshold):

    cnt_cust = 0
    accm_pct = 0

    for index, row in sales_by_cust.iterrows():
        if accm_pct ==0:
            accm_pct = row['%TotalSales']
        else:
            accm_pct = accm_pct+row['%TotalSales']
        cnt_cust = cnt_cust+1
        
        if accm_pct>=threshold:
            return cnt_cust


print("70% of sales are from ",cust_concentration(0.7)," customers")        
print("80% of sales are from ",cust_concentration(0.8)," customers")
print("90% of sales are from ",cust_concentration(0.9)," customers")
print("100% of sales are from ",sales_by_cust.CustomerID.nunique()," customers")


sales_by_prod = df[df['InvoiceDate']>dt.date(2011,9,9)].groupby('StockCode').Net.sum().sort_values(ascending=False).reset_index()
TotalSalesAll = sales_by_prod.Net.sum()
sales_by_prod['%TotalSales'] = sales_by_prod['Net']/TotalSalesAll

def prod_concentration(threshold):

    cnt_prod = 0
    accm_pct = 0

    for index, row in sales_by_prod.iterrows():
        if accm_pct ==0:
            accm_pct = row['%TotalSales']
        else:
            accm_pct = accm_pct+row['%TotalSales']
        cnt_prod = cnt_prod+1
        
        if accm_pct>=threshold:
            return cnt_prod


print("70% of sales are from ",prod_concentration(0.7)," products")        
print("80% of sales are from ",prod_concentration(0.8)," products")
print("90% of sales are from ",prod_concentration(0.9)," products")
print("100% of sales are from ",sales_by_prod.StockCode.nunique()," products")
