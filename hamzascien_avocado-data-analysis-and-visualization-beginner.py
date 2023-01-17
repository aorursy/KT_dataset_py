import pandas as pd

import numpy as np

import seaborn as sns 

import matplotlib.pyplot as plt

%matplotlib inline
data = pd.read_csv('../input/avocado-prices/avocado.csv')
data.head()
data.shape
data.info()
#missing values

data.isnull().sum()
## i will rename some columns for more flexibility

data.rename(columns={'AveragePrice':'avprice','Small Bags':'Sbags','Large Bags':'Lbags','XLarge Bags':'XLbags'},inplace=True)
data.head()
data.avprice.describe()
sns.set(style='darkgrid')

plt.figure(figsize=(11,9))

a=sns.distplot(data.avprice,color='r')

a.set_xlabel('AvragePrice')

a.set_ylabel('Frequency')

plt.title('Distribution of Average Price',size=25)

data['Total Volume'].describe()
plt.figure(figsize=(11,9))

a=sns.kdeplot(data['Total Volume'],color='g',shade=True)

a.set_xlabel('Total Volume')

a.set_ylabel('Frequency')

plt.title('Distribution of Total Volume',size=25)

data[data['Total Volume']<1500000].shape
data[data['Total Volume']>5000000].sort_values(by='Total Volume',ascending=False)
## correlation between them 

print('the correlation between AveragePrice and Total volume :',data['avprice'].corr(data['Total Volume']))
a=sns.jointplot(x='Total Volume',y='avprice',data=data,color='g',height=9)

plt.figure(figsize=(11,9))

a=sns.regplot(x='Total Volume',y='avprice',data=data[data['Total Volume']<1500000],color='c')

plt.title('Average Price vs Total Volume',size=25)
data.region.unique()
Region=data.groupby('region').avprice.mean().reset_index().sort_values(by='avprice')

Region.head()
Region.tail()
plt.figure(figsize=(11,9))

a=sns.boxplot(x='region',y='avprice',data=data,palette='nipy_spectral')

a.set_xticklabels(a.get_xticklabels(), rotation=90, ha="right",size=12)

plt.title('Boxplot AveragePrice Vs Region',size=30)
data.type.unique()
data.type.value_counts()
data.groupby('type').avprice.mean().reset_index()
a=sns.catplot(x='type',y='avprice',data=data,palette='mako',height=10,kind='boxen')

plt.title('Boxen plot of AverigePrive for each type',size=25)
a=sns.catplot(x='type',y='Total Volume',data=data[data['Total Volume']<1500000],palette='mako',height=10,kind='box')

plt.title('Boxen plot of Total Volume for each type',size=25)
sns.relplot(x="Total Volume",y='avprice',hue='type',data=data,height=10)

plt.title('AveragePrice Vs Total Volume for each type',size=25)
plt.figure(figsize=(13,15))

a=sns.barplot(x='avprice',y='region',data=data,palette='nipy_spectral',hue='type')

a.set_yticklabels(a.get_yticklabels(),size=16)

plt.title('Barplot AveragePrice Vs Region for each Type',size=30)
plt.figure(figsize=(10,15))

a=sns.barplot(x='Total Volume',y='region',data=data[data.type=='organic'].query('region != "TotalUS"'),palette='coolwarm')

a.set_yticklabels(a.get_yticklabels(),size=16)

plt.title('Total Volume for organic for each Region',size=30)
plt.figure(figsize=(10,15))

a=sns.barplot(x='Total Volume',y='region',data=data[data.type=='conventional'].query('region != "TotalUS"'),palette='coolwarm')

a.set_yticklabels(a.get_yticklabels(),size=16)

plt.title('Total Volume for conventional for each Region',size=30)
data.year.unique()
data.Date=pd.to_datetime(data.Date)
data['month']=data.Date.dt.month
price_years=data.groupby(['year','month','type'],as_index=False)['avprice'].mean()

price_years
plt.figure(figsize=(13,9))

a=sns.lineplot(x='month',y='avprice',data=price_years.query('year=="2015"')

            ,hue='type',markers=True,style='type'

            ,palette='gnuplot2' )
plt.figure(figsize=(13,9))

a=sns.lineplot(x='month',y='avprice',data=price_years.query('year=="2016"')

            ,hue='type',markers=True,style='type'

            ,palette='gnuplot2' )
plt.figure(figsize=(13,9))

a=sns.lineplot(x='month',y='avprice',data=price_years.query('year=="2017"')

            ,hue='type',markers=True,style='type'

            ,palette='gnuplot2' )
plt.figure(figsize=(13,9))

a=sns.lineplot(x='month',y='avprice',data=price_years.query('year=="2018"')

            ,hue='type',markers=True,style='type'

            ,palette='gnuplot2' )
sns.factorplot('avprice','region',data=data.query("type=='conventional'"),

                hue='year',

                size=15,

                palette='tab20',

                join=False,

                aspect=0.7,

              )

plt.title('For Conventional',size=25)
sns.factorplot('avprice','region',data=data.query("type=='organic'"),

                hue='year',

                size=15,

                palette='tab20',

                join=False,

                aspect=0.7,

              )

plt.title('For Organic',size=25)