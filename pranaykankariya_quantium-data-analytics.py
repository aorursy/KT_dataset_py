import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

plt.style.use('ggplot')

import seaborn as sns

sns.set_palette("rocket_r")

import datetime as dt

import re

%matplotlib inline
transaction = pd.read_csv("../input/quantium-data-analytics-virtual-experience-program/Transactions.csv")

behaviour = pd.read_csv("../input/quantium-data-analytics-virtual-experience-program/PurchaseBehaviour.csv")
transaction.head()
transaction.info()
behaviour.head()
behaviour.info()
transaction.describe()
transaction[transaction.PROD_QTY>100]
#Dropping rows from the main data

transaction  = transaction.drop([69762,69763],axis=0).reset_index(drop=True)
# Brand Column

transaction['BRAND'] = transaction['PROD_NAME'].apply(lambda x : x.strip().split()[0])

brands = {'Dorito':'Doritos','Infzns':'Infuzions','Snbts':'Sunbites','Grain':'Grain Wave','Red':'RRD','Smith':'Smiths','GrnWves':'Grain Wave','ww':'Woolworths','NCC':'Natural'}

transaction['BRAND'] = transaction['BRAND'].map(brands).fillna(transaction['BRAND'])
#Packet Size

def get_size(item):

    size=[]

    for i in item:

        if i.isdigit():

            size.append(i)

    return int("".join(size))



transaction['PACKET_SIZE'] = transaction['PROD_NAME'].apply(lambda x : get_size(x))
#Product Unit Price

transaction['PROD_UNIT_PRICE'] = transaction['TOT_SALES']/transaction['PROD_QTY']
#Salsa

transaction[transaction['PROD_NAME'].str.contains('salsa',case=False)]['PROD_NAME'].unique()
salsa_index = transaction.index[(transaction.PACKET_SIZE==300) & transaction['PROD_NAME'].str.contains('salsa',case=False) & ~transaction['PROD_NAME'].str.contains('Doritos',case=False)]



#Drop rows from the main data

transaction = transaction.drop(salsa_index,axis=0,).reset_index(drop=True)
# Counts of different brands according to packet size

transaction.groupby(['BRAND'])['PACKET_SIZE'].value_counts()
# Brands Plot

fig,ax = plt.subplots(figsize=(20,18))

plt.subplot(3,1,1)

sns.countplot(transaction['BRAND'],palette='rocket_r').set(ylabel='TRANSACTIONS')

plt.title("Transactions of different Brands")



# Packet Size plot

plt.subplot(3,1,2)

sns.countplot(transaction['PACKET_SIZE'],palette='rocket_r').set(ylabel='TRANSACTIONS')

plt.title("Transactions of different Packet Sizes")



fig.suptitle("Analysing PROD_NAME",fontsize=20)
sns.distplot(transaction['PROD_UNIT_PRICE'])
# Date

transaction['DATE'] = pd.to_datetime(transaction['DATE'],unit='D',origin=pd.Timestamp("30-12-1899")).dt.normalize()

transaction['YEAR'] = pd.DatetimeIndex(transaction['DATE']).year

transaction['MONTH'] = pd.DatetimeIndex(transaction['DATE']).month
fig,ax = plt.subplots(figsize=(20,12))

plt.subplot(2,2,1)

sns.countplot(transaction['YEAR']).set(ylabel='TRANSACTIONS')

plt.title("Transaction per Year")

plt.subplot(2,2,2)

sns.countplot(transaction['MONTH'],palette='rocket_r').set(ylabel='TRANSACTIONS')

plt.title("Transactions per Month")

plt.subplot(2,2,3)

sns.countplot(transaction[transaction['YEAR']==2018]['MONTH']).set(ylabel='TRANSACTIONS')

plt.title("Transactions for the Year 2018")

plt.subplot(2,2,4)

sns.countplot(transaction[transaction['YEAR']==2019]['MONTH']).set(ylabel='TRANSACTIONS')

plt.title("Transactions for the Year 2019")
#Total sales - date

date_sale = transaction.groupby('DATE').agg({'TOT_SALES':'sum'}).reset_index()

plt.figure(figsize=(20,6))

sns.lineplot(x=date_sale['DATE'],y=date_sale['TOT_SALES'])
fig,ax = plt.subplots(figsize=(20,12))

plt.subplot(2,1,1)

sns.countplot(transaction['STORE_NBR'],order = transaction['STORE_NBR'].value_counts().head(50).index,palette='rocket_r').set(ylabel='TRANSACTIONS')

plt.title("First 50 Stores with highest transactions")

plt.subplot(2,1,2)

sns.countplot(transaction['STORE_NBR'],order = transaction['STORE_NBR'].value_counts().tail(50).index,palette='rocket_r').set(ylabel='TRANSACTIONS')

plt.title("Last 50 Stores with lowest transactions")

print("Max:",transaction['STORE_NBR'].value_counts().max())

print("Min:",transaction['STORE_NBR'].value_counts().min())
fig,ax = plt.subplots(figsize=(25,12))

plt.subplot(2,1,1)

sns.countplot(transaction['PROD_NBR'],order=transaction['PROD_NBR'].value_counts().head(50).index,palette='rocket_r').set(ylabel='TRANSACTIONS')

plt.title("First 50 Products with highest transaction")

plt.subplot(2,1,2)

sns.countplot(transaction['PROD_NBR'],order=transaction['PROD_NBR'].value_counts().tail(50).index,palette='rocket_r').set(ylabel='TRANSACTIONS')

plt.title("Last 50 Products with lowest transaction")



print("Max:",transaction['PROD_NBR'].value_counts().max())

print("Min:",transaction['PROD_NBR'].value_counts().min())
sns.countplot(behaviour['PREMIUM_CUSTOMER'])
plt.figure(figsize=(20,5))

sns.countplot(behaviour['LIFESTAGE'],palette='rocket_r')
plt.figure(figsize=(20,5))

sns.countplot(behaviour['LIFESTAGE'],palette='rocket_r',hue=behaviour['PREMIUM_CUSTOMER'])
combined = transaction.merge(behaviour,how='left',on='LYLTY_CARD_NBR')

combined = combined[['DATE','YEAR','MONTH','STORE_NBR','TXN_ID','PROD_NBR','PROD_NAME','BRAND','PACKET_SIZE',

                      'PROD_QTY','PROD_UNIT_PRICE','TOT_SALES','LYLTY_CARD_NBR','LIFESTAGE','PREMIUM_CUSTOMER']]

combined.head()
customer_group = combined.groupby(['LIFESTAGE','PREMIUM_CUSTOMER']).agg({'TOT_SALES':'sum','PROD_QTY':'sum','LYLTY_CARD_NBR':'count'}).reset_index()

customer_group['SEGMENT'] = customer_group.LIFESTAGE + '-' + customer_group.PREMIUM_CUSTOMER.apply(lambda x:x.upper())

customer_group['SALES_PER_CUSTOMER'] = customer_group['TOT_SALES']/customer_group['LYLTY_CARD_NBR']
fig,ax = plt.subplots(figsize=(20,30))

plt.subplot(3,1,1)

customer_group = customer_group.sort_values('TOT_SALES')

sns.barplot(y='SEGMENT',x='TOT_SALES',data=customer_group,orient='h',palette='rocket_r')

plt.title("Total Sales of different segments")



plt.subplot(3,1,2)

customer_group = customer_group.sort_values('PROD_QTY')

sns.barplot(y='SEGMENT',x='PROD_QTY',data=customer_group,orient='h',palette='rocket_r')

plt.title("Products Purched by different segments")



plt.subplot(3,1,3)

customer_group = customer_group.sort_values('SALES_PER_CUSTOMER')

sns.barplot(y='SEGMENT',x='SALES_PER_CUSTOMER',data=customer_group,orient='h',palette='rocket_r')

plt.title("Units Per Customer")
young_main = combined[(combined['LIFESTAGE']=='YOUNG SINGLES/COUPLES') & (combined['PREMIUM_CUSTOMER']=='Mainstream')]

other = combined[(combined['LIFESTAGE']!='YOUNG SINGLES/COUPLES') & (combined['PREMIUM_CUSTOMER']!='Mainstream')]
plt.subplots(figsize=(20,25))

plt.subplot(4,1,1)

sns.barplot(x='BRAND',y='TOT_SALES',data=young_main,palette='rocket_r')

plt.title("Mainstream Young Single Couple - Brand vs Total Sales")

plt.subplot(4,1,2)

sns.barplot(x='BRAND',y='TOT_SALES',data=other,palette='rocket_r')

plt.title("Other Segments - SIZE vs Total Sales")

plt.subplot(4,1,3)

sns.barplot(x='PACKET_SIZE',y='TOT_SALES',data=young_main,palette='rocket_r')

plt.title("Mainstream Young Single Couple - Size vs Total Sales")

plt.subplot(4,1,4)

sns.barplot(x='PACKET_SIZE',y='TOT_SALES',data=other,palette='rocket_r')

plt.title("Other Segments - Size vs Total Sales")