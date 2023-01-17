import os

print(os.getcwd())

import matplotlib.pyplot as plt

import pandas as pd

import numpy as np
df=pd.read_csv('../input/vgsales.csv')

df['Year']=df['Year'].astype(object) #year as object instead float

df['Rank']=df['Rank'].astype(object) #rank as object instead int

print(df.dtypes)

#df=df[df['Year']==2005]

df.head()
var=df.groupby(['Name']).sum()

var.sort_values(inplace=True,ascending=False,by='Global_Sales')

plt.close()

fig=plt.figure()

ax = fig.add_subplot(1,1,1) # Create matplotlib axes

var['Global_Sales'].head(10).plot(kind='barh',stacked=True,ax=ax,color='sienna',position=0,width=0.3,title='Sales by Game\'s Name')

var[['EU_Sales','NA_Sales','JP_Sales','Other_Sales']].head(10).plot(kind='barh',stacked=True,ax=ax,position=1,width=0.3)

plt.show()
var=df.groupby(['Genre']).sum()

var.sort_values(inplace=True,ascending=False,by='Global_Sales')

fig=plt.figure()

ax = fig.add_subplot(1,1,1) # Create matplotlib axes

ax2 = ax.twinx()

var['Global_Sales'].plot(kind='bar',ax=ax,position=0,color='sienna',width=0.3,legend=True,title='Sales by Genere')

var[['NA_Sales','EU_Sales','JP_Sales','Other_Sales']].plot(kind='bar',stacked=True,ax=ax2,position=1,width=0.3)

plt.show()

plt.close()
var=df.groupby(['Year']).sum()

var.plot(kind='line',title='Sales by Year')

plt.show()

plt.close()
var=df.groupby(['Publisher']).sum()

var.sort_values(inplace=True,ascending=False,by='Global_Sales')

plt.close()

fig=plt.figure()

ax = fig.add_subplot(1,1,1) # Create matplotlib axes

var['Global_Sales'].head(10).plot(kind='barh',stacked=True,ax=ax,color='sienna',position=0,width=0.3,title='Sales By Publisher')

var[['EU_Sales','NA_Sales','JP_Sales','Other_Sales']].head(10).plot(kind='barh',stacked=True,ax=ax,position=1,width=0.3)

plt.show()
sales=['Global_Sales','NA_Sales','EU_Sales','JP_Sales','Other_Sales']

from scipy.stats import pearsonr

for i in range(len(sales)):

    for j in range(i+1,len(sales)):

        print(sales[i] + "," + sales[j] +"=" + str(pearsonr(df[sales[i]],df[sales[j]])))
var=df.groupby(['Platform']).sum()

var.sort_values(inplace=True,ascending=False,by='Global_Sales')

plt.close()

fig=plt.figure()

ax = fig.add_subplot(1,1,1) # Create matplotlib axes

var['Global_Sales'].head(10).plot(kind='barh',stacked=True,ax=ax,color='sienna',position=0,width=0.3,title='Sales By Platform')

var[['EU_Sales','NA_Sales','JP_Sales','Other_Sales']].head(10).plot(kind='barh',stacked=True,ax=ax,position=1,width=0.3)

plt.show()
var=df.groupby(['Platform']).agg(['count','sum'])

var.sort_values(inplace=True,by=('Name','count'),ascending=False)

var[['Name','Global_Sales']].head()

plt.close()

fig=plt.figure()

ax = fig.add_subplot(1,1,1) # Create matplotlib axes

ax.set_xlabel('Number of Games Available')

ax2 = ax.twiny() # Create matplotlib axes

ax2.set_xlabel('Total_Global_Sales')

var['Name']['count'].head(10).plot(kind='barh',stacked=True,ax=ax,color='sienna',position=1,width=0.3)

var['Global_Sales']['sum'].head(10).plot(kind='barh',stacked=True,ax=ax2,position=0,width=0.3,legend=True)

plt.show()
plt.close()

var=df.groupby(['Year','Genre'])#.groupby(['Year','Genre'])['Global_Sales'].max()

x=np.unique(df['Year'].dropna())

y=np.unique(df['Genre'].dropna())

maxGlobalSalesbyYear = [('Year','Sales')]

for i,j in var:

    k=j['Global_Sales'].sum()

    maxGlobalSalesbyYear.append((i[0],(k,i[1])))

headers=maxGlobalSalesbyYear.pop(0)

df2=pd.DataFrame(maxGlobalSalesbyYear,columns=headers)

df3=df2.groupby('Year').max()

df3[['T_Sales','Genre']]=df3['Sales'].apply(pd.Series)

df3['Year']=x

del df3['Sales']

print("YearWise Best Genre")

df3['Genre']
df2=df[df['Publisher']=='Electronic Arts'].groupby('Year')['Global_Sales'].sum()

df3=df[df['Publisher']=='Nintendo'].groupby(['Year'])['Global_Sales'].sum()

var=pd.DataFrame()

var['EA']=df2

var['Nintendo']=df3

var.plot(kind='line')

plt.title('Sales comparision EA vs Nintendo')

plt.show()
var=df[df['Year']==2015].groupby(['Name'])['Global_Sales'].sum()

var.sort_values(inplace=True,ascending=False)

plt.close()

var.head(10).plot.pie(figsize=(5,5),autopct='%.2f')

plt.title('Top 10 games of 2015 by Global Sales')

plt.show()