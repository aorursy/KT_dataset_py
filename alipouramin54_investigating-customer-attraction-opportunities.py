import numpy as np

import pandas as pd

import datetime as dt

import matplotlib.pyplot as plt

%matplotlib inline
df1=pd.read_excel('/kaggle/input/uci-online-retail-ii-data-set/online_retail_II.xlsx',sheet_name='Year 2009-2010')

df2=pd.read_excel('/kaggle/input/uci-online-retail-ii-data-set/online_retail_II.xlsx',sheet_name='Year 2010-2011')
online_retail_II=pd.concat([df1,df2])
online_retail_II=online_retail_II.reset_index(drop=True)
online_retail_II.head()
online_retail_II['Revenue']=online_retail_II['Quantity']*online_retail_II['Price']
online_retail_II.info()
online_retail_II.isna().sum()
online_retail_II.shape
print("%",100*(online_retail_II['Customer ID'].isna().sum()/online_retail_II.shape[0]))
online_retail_II_1=online_retail_II.dropna()
online_retail_II_1.head()
online_retail_II_1=online_retail_II_1.reset_index(drop=True)
df3=online_retail_II_1.drop_duplicates(subset=['Invoice']).reset_index(drop=True)

df3.head()
UC=pd.DataFrame(online_retail_II['Customer ID'].unique()).dropna().count()[0] #UC = Unique Customer ID

print('Total number of customers: ',UC)
BUC=df3['Customer ID'].value_counts().reset_index()

NBUC=BUC[BUC['Customer ID']==1].count()[0]

print('Percentage of customers who have purchased only once: ','%',(NBUC/UC)*100)
item=2

while item<BUC['Customer ID'].max():

    NBUC1=BUC[BUC['Customer ID']==item].count()[0]

    if ((NBUC1/UC)*100)>=1:

        print('Percentage of customers who have purchased ',item,' times: ','%',(NBUC1/UC)*100)

    item+=1

print('The remaining percentages are less than 1 percent')

print('The most quantity of purchases = ',BUC['Customer ID'].max())
BUC.set_axis(['Customer ID','Frequency of Purchases'],inplace=True,axis=1)

C1P = BUC[BUC['Frequency of Purchases']==1].sort_values('Customer ID').reset_index(drop=True)

C1P.head() #C1P: Customers with one purchase
df_C1P = pd.merge(online_retail_II,C1P['Customer ID'],how='inner').sort_values('Customer ID').reset_index(drop=True)

df_C1P.head()
df_FS = df_C1P['StockCode'].value_counts().reset_index()

df_FS.set_axis(['StockCode','Frequency'],inplace=True,axis=1)

df_FS.head()
df_FS = pd.merge(df_FS,df_C1P[['StockCode','Description']],how='left').drop_duplicates(subset=['StockCode']).reset_index(drop=True)

df_FS = df_FS[['StockCode','Description','Frequency']]

df_FS.head()
df_ParetoF = df_FS

df_ParetoF['cumpercentage_F'] = df_ParetoF['Frequency'].cumsum()/df_ParetoF['Frequency'].sum()

df_ParetoF = df_ParetoF[df_ParetoF['cumpercentage_F']<=0.70]

df_ParetoF.head()
df_RS = df_C1P.groupby('StockCode').sum()['Revenue'].sort_values(ascending=False).reset_index()

df_RS.set_axis(['StockCode','Revenue'],inplace=True,axis=1)

df_RS.head()
df_RS = pd.merge(df_RS,df_C1P[['StockCode','Description']],how='right').drop_duplicates(subset=['StockCode']).reset_index(drop=True)

df_RS = df_RS[['StockCode','Description','Revenue']]

df_RS.head()
df_ParetoR = df_RS

df_ParetoR['cumpercentage_R'] = df_ParetoR['Revenue'].cumsum()/df_ParetoR['Revenue'].sum()

df_ParetoR = df_ParetoR[df_ParetoR['cumpercentage_R']<=0.70]

df_ParetoR.head()
df_result = pd.merge(df_ParetoF.drop(columns=['cumpercentage_F']),df_ParetoR[['StockCode','Description','Revenue']],how='inner').drop_duplicates(subset=['StockCode']).reset_index(drop=True)

df_result.head()
df_C1P['InvoiceYearMonth']=pd.to_datetime(df_C1P['InvoiceDate']).map(lambda date: str((date.year))+'-'+dt.datetime(2000,date.month,29).strftime('%m'))

df_C1P.head()
df_MonthlyNewCustomer = df_C1P.groupby('InvoiceYearMonth')['Customer ID'].nunique().reset_index()

df_MonthlyNewCustomer.set_axis(['InvoiceYearMonth','Number of New Customers'],inplace=True,axis=1)

df_MonthlyNewCustomer.head()
fig = plt.figure()

axes = fig.add_axes([0, 0, 3, 1])

axes.bar(df_MonthlyNewCustomer['InvoiceYearMonth'],height=df_MonthlyNewCustomer['Number of New Customers'],color="Green")

axes.set_xlabel('Date',size=20)

axes.set_ylabel('Number',size=20)

axes.set_title('Monthly New Customer',size=24);

plt.show()