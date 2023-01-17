import numpy as np 

import pandas as pd 

%matplotlib inline

import matplotlib.pyplot as plt

import seaborn as sns

from mlxtend.preprocessing import TransactionEncoder

from mlxtend.frequent_patterns import apriori, association_rules

import networkx as nx

import warnings

warnings.filterwarnings('ignore')
data=pd.read_csv('../input/transactions-from-a-bakery/BreadBasket_DMS.csv')

data.head()
data.shape
data.describe()
data.shape
data.info()
data.loc[data['Item']=='NONE',:].count()
data=data.drop(data.loc[data['Item']=='NONE'].index)
data['Item'].nunique()
data['Item'].value_counts().sort_values(ascending=False).head(10)
fig, ax=plt.subplots(figsize=(6,4))

data['Item'].value_counts().sort_values(ascending=False).head(10).plot(kind='bar')

plt.ylabel('Number of transactions')

plt.xlabel('Items')

ax.get_yaxis().get_major_formatter().set_scientific(False)

plt.title('Best sellers')
data.loc[(data['Time']<'12:00:00'),'Daytime']='Morning'

data.loc[(data['Time']>='12:00:00')&(data['Time']<'17:00:00'),'Daytime']='Afternoon'

data.loc[(data['Time']>='17:00:00')&(data['Time']<'21:00:00'),'Daytime']='Evening'

data.loc[(data['Time']>='21:00:00')&(data['Time']<'23:50:00'),'Daytime']='Night'
fig, ax=plt.subplots(figsize=(6,4))

sns.set_style('darkgrid')

data.groupby('Daytime')['Item'].count().sort_values().plot(kind='bar')

plt.ylabel('Number of transactions')

ax.get_yaxis().get_major_formatter().set_scientific(False)

plt.title('Business during the day')
data.groupby('Daytime')['Item'].count().sort_values(ascending=False)
data['Date_Time']=pd.to_datetime(data['Date']+' '+data['Time'])

data['Day']=data['Date_Time'].dt.day_name()

data['Month']=data['Date_Time'].dt.month

data['Month_name']=data['Date_Time'].dt.month_name()

data['Year']=data['Date_Time'].dt.year

data['Year_Month']=data['Year'].apply(str)+' '+data['Month_name'].apply(str)

data.drop(['Date','Time'], axis=1, inplace=True)



data.index=data['Date_Time']

data.index.name='Date'

data.drop(['Date_Time'],axis=1,inplace=True)

data.head()
data.groupby('Year_Month')['Item'].count().plot(kind='bar')

plt.ylabel('Number of transactions')

plt.title('Business during the past months')
data.loc[data['Year_Month']=='2016 October'].nunique()
data.loc[data['Year_Month']=='2017 April'].nunique()
data2=data.pivot_table(index='Month_name',columns='Item', aggfunc={'Item':'count'}).fillna(0)

data2['Max']=data2.idxmax(axis=1)

data2

data3=data.pivot_table(index='Daytime',columns='Item', aggfunc={'Item':'count'}).fillna(0)

data3['Max']=data3.idxmax(axis=1)

data3
data4=data.pivot_table(index='Day',columns='Item', aggfunc={'Item':'count'}).fillna(0)

data4['Max']=data4.idxmax(axis=1)

data4
data['Item'].resample('M').count().plot()

plt.ylabel('Number of transactions')

plt.title('Business during the past months')
data['Item'].resample('W').count().plot()

plt.ylabel('Number of transactions')

plt.title('Weekly business during the past months')
data['Item'].resample('D').count().plot()

plt.ylabel('Number of transactions')

plt.title('Daily business during the past months')
data['Item'].resample('D').count().min()
data['Item'].resample('D').count().max()
lst=[]

for item in data['Transaction'].unique():

    lst2=list(set(data[data['Transaction']==item]['Item']))

    if len(lst2)>0:

        lst.append(lst2)

print(lst[0:3])

print(len(lst))
te=TransactionEncoder()

te_data=te.fit(lst).transform(lst)

data_x=pd.DataFrame(te_data,columns=te.columns_)

print(data_x.head())



frequent_items= apriori(data_x, use_colnames=True, min_support=0.03)

print(frequent_items.head())



rules = association_rules(frequent_items, metric="lift", min_threshold=1)

rules.antecedents = rules.antecedents.apply(lambda x: next(iter(x)))

rules.consequents = rules.consequents.apply(lambda x: next(iter(x)))

rules
fig, ax=plt.subplots(figsize=(10,4))

GA=nx.from_pandas_edgelist(rules,source='antecedents',target='consequents')

nx.draw(GA,with_labels=True)

plt.show()

import pandas as pd

BreadBasket_DMS = pd.read_csv("../input/transactions-from-a-bakery/BreadBasket_DMS.csv")