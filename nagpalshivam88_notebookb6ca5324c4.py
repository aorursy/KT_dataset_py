# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

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
import pandas as pd
pd.set_option('display.max_rows', 50)
pd.set_option('display.max_columns', 150)
import matplotlib.pyplot as plt
from datetime import datetime as dt
import numpy as np
df = pd.read_csv('../input/postman/sales_data.csv',encoding='latin1')
df['Date'] = pd.to_datetime(df['date'], format='%m/%d/%Y')
df.head(-5)
#Getting Month-Year from Order Date
df['Month_Year'] = df['Date'].apply(lambda x: x.strftime('%Y-%m'))
df['Sales'] = df['unit price']*df['quantity sold']
results = df.groupby('Month_Year').sum()
results.head()
months = [month for month, df in df.groupby('Month_Year')]
plt.figure(figsize=(15,5))
plt.plot(months,results['Sales'], color = '#b80045')
plt.xticks(months, rotation='vertical', size = 8)
plt.ylabel('Sales in USD')
plt.xlabel('Month')
#plt.grid()
plt.show()
MoM_Data = pd.DataFrame(results['Sales'])
MoM_Data['Last_Month'] = np.roll(MoM_Data['Sales'],1)
MoM_Data
#Now, since the MoM first month cannot be found since we donot have it's previous month value.
#By default, python has put the last value in the column and rolled up over there. SO we need to remove that.

MoM_Data = MoM_Data.drop(MoM_Data.index[0])
MoM_Data['Growth'] = (MoM_Data['Sales']/MoM_Data['Last_Month'])-1
MoM_Data.head()
results = MoM_Data.drop(columns = ["Sales", "Last_Month"])
results['Months'] = results.index
results.reset_index(drop=True, inplace=True)
results.head()
plt.figure(figsize=(15,5))
plt.bar(results['Months'],results['Growth']*100, color = '#b80045')
plt.xticks(results['Months'], rotation='vertical', size = 8)
plt.ylabel('% Growth')
plt.xlabel('Month')
plt.title("\n MoM Growth Over Time \n", size=25)
#plt.grid()
plt.show()
df['Qtr'] = df['Date'].apply(lambda x: x.strftime('%m'))
df['Qtr'] = pd.to_numeric(df['Qtr'])//4+1
df['Year'] = df['Date'].apply(lambda x: x.strftime('%Y'))
df['Qtr_Yr'] = df['Year'].astype(str) + '-Q' + df['Qtr'].astype(str)
df.drop('Qtr', axis=1)
df.head()
results = df.groupby('Qtr_Yr').sum()
results.head()
QoQ_Data = pd.DataFrame(results['Sales'])

QoQ_Data['Last_Qtr'] = np.roll(QoQ_Data['Sales'],1)
QoQ_Data
#Now, since the MoM first month cannot be found since we donot have it's previous month value.
#By default, python has put the last value in the column and rolled up over there. SO we need to remove that.

QoQ_Data = QoQ_Data.drop(QoQ_Data.index[0])
#Calculating QoQ Growth for each Qtr:
QoQ_Data['Growth'] = (QoQ_Data['Sales']/QoQ_Data['Last_Qtr'])-1
QoQ_Data.head()
##Plotting QoQ Growth
results = QoQ_Data.drop(columns = ["Sales", "Last_Qtr"])
results['Quarter'] = results.index
results.reset_index(drop=True, inplace=True)
results.head()
plt.figure(figsize=(15,5))
plt.bar(results['Quarter'],results['Growth']*100, color = '#b80045')
plt.xticks(results['Quarter'], rotation='vertical', size = 8)
plt.ylabel('% Growth')
plt.xlabel('\n Quarter')
plt.title("\n QoQ Growth Over Time \n", size=25)
#plt.grid()
plt.show()
#Creating sales grouped by Month-Year again
YoY_Data = pd.DataFrame(df.groupby('Month_Year').sum()['Sales'])
YoY_Data['Last_Year'] = np.roll(YoY_Data['Sales'],12)
#YoY_Data
YoY_Data = YoY_Data.drop(YoY_Data.index[0:12])
YoY_Data.head()

#Calculating YoY Growth for each month:
YoY_Data['Growth'] = (YoY_Data['Sales']/YoY_Data['Last_Year'])-1
YoY_Data.head()
##Plotting YoY Growth
results = YoY_Data.drop(columns = ["Sales", "Last_Year"])
results['Month_Year'] = results.index
results.reset_index(drop=True, inplace=True)
results.head()
plt.figure(figsize=(15,5))
plt.bar(results['Month_Year'],results['Growth']*100, color = '#b80045')
plt.xticks(results['Month_Year'], rotation='vertical', size = 8)
plt.ylabel('% Growth')
plt.xlabel('\n Month')
plt.title("\n YoY Growth Over Months \n", size=25)
#plt.grid()
plt.show()

#Q3. Which are the Top 10 products by sales?¶
prod_sales = pd.DataFrame(df.groupby('product id').sum()['Sales'])
prod_sales.sort_values(by=['Sales'], inplace=True, ascending=False)

#Calculating Top 10:
top_prods = prod_sales.head(10)
top_prods
#Q4. Which are the most selling products?
best_selling_prods = pd.DataFrame(df.groupby('product id').sum()['quantity sold'])
best_selling_prods.sort_values(by=['quantity sold'], inplace=True, ascending=False)

#Calculating Top 5:
best_selling_prods = best_selling_prods.head(5)
best_selling_prods
#how many orders have they made
invoice_ct = df.groupby(by='customer id', as_index=False)['transaction id'].count()
invoice_ct.columns = ['CustomerID', 'NumberOrders']
invoice_ct.head()
#remove the negative values and replace with nan
import numpy as np
df[df['quantity sold'] < 0] = np.nan
df[df['unit price'] < 0] = np.nan
df.describe()
#how much money have they spent
total_spend = df.groupby(by='customer id', as_index=False)['Sales'].sum()
total_spend.columns = ['CustomerID', 'total_spent']
total_spend.describe()
#how many items they bought
total_items = df.groupby(by='customer id', as_index=False)['quantity sold'].sum()
total_items.columns = ['CustomerID', 'NumberItems']
total_items.describe()
#when was their first order and how long ago was that from the last date in file (presumably
#when the data were pulled)
earliest_order = df.groupby(by='customer id', as_index=False)['Date'].min()
earliest_order.columns = ['CustomerID', 'EarliestInvoice']
earliest_order['now'] = pd.to_datetime((df['Date']).max())
earliest_order['days_as_customer'] = 1 + (earliest_order.now-earliest_order.EarliestInvoice).astype('timedelta64[D]')
earliest_order.drop('now', axis=1, inplace=True)
earliest_order
#when was their last order and how long ago was that from the last date in file (presumably
#when the data were pulled)
last_order = df.groupby(by='customer id', as_index=False)['Date'].max()
last_order.columns = ['CustomerID', 'last_purchase']
last_order['now'] = pd.to_datetime((df['Date']).max())
last_order['days_since_purchase'] = 1 + (last_order.now-last_order.last_purchase).astype('timedelta64[D]')
last_order.drop('now', axis=1, inplace=True)
last_order.head
#combine all the dataframes into one
import functools
dfs = [total_spend,invoice_ct,earliest_order,last_order,total_items]
CustomerTable = functools.reduce(lambda left,right: pd.merge(left,right,on='CustomerID', how='outer'), dfs)
CustomerTable['Recency'] = CustomerTable['days_as_customer']-CustomerTable['days_since_purchase']
CustomerTable.describe()

df_RFM = df.groupby('customer id').agg({'Date': lambda y: (df['Date'].max().date() - y.max().date()).days,
                                        'transaction id': lambda y: len(y.unique()),  
                                        'Sales': lambda y: round(y.sum(),2)})
df_RFM.columns = ['Recency', 'Frequency', 'Monetary']
df_RFM = df_RFM.sort_values('Monetary', ascending=False)
df_RFM.head()
# We will use the 80% quantile for each feature
quantiles = df_RFM.quantile(q=[0.8])
print(quantiles)
df_RFM['R']=np.where(df_RFM['Recency']<=int(quantiles.Recency.values), 2, 1)
df_RFM['F']=np.where(df_RFM['Frequency']>=int(quantiles.Frequency.values), 2, 1)
df_RFM['M']=np.where(df_RFM['Monetary']>=int(quantiles.Monetary.values), 2, 1)
df_RFM.head()
# To do the 2 x 2 matrix we will only use Recency & Monetary
df_RFM['RMScore'] = df_RFM.M.map(str)+df_RFM.R.map(str)
df_RFM = df_RFM.reset_index()
df_RFM_SUM = df_RFM.groupby('RMScore').agg({'customer id': lambda y: len(y.unique()),
                                        'Frequency': lambda y: round(y.mean(),0),
                                        'Recency': lambda y: round(y.mean(),0),
                                        'R': lambda y: round(y.mean(),0),
                                        'M': lambda y: round(y.mean(),0),
                                        'Monetary': lambda y: round(y.mean(),0)})
df_RFM_SUM = df_RFM_SUM.sort_values('RMScore', ascending=False)
df_RFM_SUM.head()
# 1) Average Monetary Matrix
df_RFM_M = df_RFM_SUM.pivot(index='M', columns='R', values='Monetary')
df_RFM_M= df_RFM_M.reset_index().sort_values(['M'], ascending = False).set_index(['M'])
df_RFM_M


# 2) Number of Customer Matrix
df_RFM_C = df_RFM_SUM.pivot(index='M', columns='R', values='customer id')
df_RFM_C= df_RFM_C.reset_index().sort_values(['M'], ascending = False).set_index(['M'])
df_RFM_C
# 3) Recency Matrix
df_RFM_R = df_RFM_SUM.pivot(index='M', columns='R', values='Recency')
df_RFM_R= df_RFM_R.reset_index().sort_values(['M'], ascending = False).set_index(['M'])
df_RFM_R
