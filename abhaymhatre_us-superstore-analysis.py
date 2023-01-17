import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
df = pd.read_excel('/kaggle/input/superstore/US Superstore data.xls')
df.head()
df.isnull().sum()
df.City.nunique()
df.City.value_counts()
len(df.State.value_counts())
df.State.value_counts()
df.Category.value_counts().plot(kind='bar')
plt.xticks(rotation=0)
plt.show()
pd.crosstab( df['Sub-Category'],df['Category'])
pd.crosstab( df['Region'],df['Category']).plot(kind='bar',stacked='True', figsize=(10,7))
plt.xticks(rotation=0)
plt.plot()
pd.crosstab( df['Region'],df['Category']).plot(kind='bar', figsize=(10,7))
plt.xticks(rotation=0)
plt.plot()
((df.groupby(['Region','Category']).count()['Row ID'])/(df.groupby('Region').count()['Row ID'])*100).plot(kind='bar')
((df.groupby(['Region','Category']).count()['Row ID'])/(df.groupby('Region').count()['Row ID'])*100).plot(kind='bar')
# adding %ge profit column:
df['% profit'] = df['Profit']/df['Sales']*100
df.head()
max(df['% profit'])
min(df['% profit'])
loss = df[df['% profit']<0]
loss
df.info
1871/9994*100
# city/state/region wise profit

profit_city = df.groupby('City').sum()['Profit']
profit_city
# top 5 cities with highest profit
profit_city.sort_values(ascending=False).head()
# top 5 cities with lowest profit
profit_city.sort_values(ascending=True).head()
fig, axes = plt.subplots(1,2, figsize=(15,5))
plt.xticks(rotation=45)
profit_city.sort_values(ascending=False).head().plot(kind='bar', ax=axes[0])
# plt.title('Maximum Profit')
plt.legend()
profit_city.sort_values(ascending=True).head().plot(kind='bar', ax=axes[1])
# plt.setp(axes.get_xticklabels(), rotation=45)
# plt.xticks(rotation=45)
# plt.title('Minimum Profit')
plt.show()
# state wise profit distribution
state_profit = df.groupby('State').sum()['Profit']
state_profit
plt.figure(figsize=(15,5))
sns.barplot(x=state_profit.index, y=state_profit.values)
plt.xticks(rotation=90)

plt.show()
state_profit.sort_values(ascending=False)
df_state_profit = pd.DataFrame(state_profit)
df_state_profit.head()
df_state_profit['% Contribution'] = df_state_profit['Profit']/sum(df_state_profit['Profit'])*100
df_state_profit
sum(df_state_profit['Profit'])
# region wise profit:
region_profit = pd.DataFrame(df.groupby('Region').sum()['Profit'])
region_profit
plt.pie(region_profit, labels=region_profit.index, autopct='%.2f', explode=(0,0,0,0.1), shadow=True)
plt.show()
# city wise sale:
city_sale = pd.DataFrame(df.groupby('City').sum()['Sales'])

# state wise sale
state_sale = pd.DataFrame(df.groupby('State').sum()['Sales'])

# region wise sale
region_sale = pd.DataFrame(df.groupby('Region').sum()['Sales'])
# maximum sale city wise
city_sale.sort_values(by='Sales',ascending=False).head(10)
# top 10 cities with lowest sales:
city_sale.sort_values(by='Sales',ascending=True).head(10)
# maximum sale w.r.t. states:
state_sale.sort_values(by='Sales',ascending=False).head(5)
# lowest sales w.r.t. states:
state_sale.sort_values(by='Sales',ascending=True).head(5)
# maximum and minmum sale w.r.t. region:
region_sale.sort_values(by='Sales', ascending=True)
df[df['Discount']==df['Discount'].max()].groupby('Product Name').sum()['Quantity'].sort_values(ascending=False)
df[df['Discount']==df['Discount'].max()]
df['Discount'].max()

# most popular product:

df.groupby('Product Name').sum()['Quantity'].sort_values(ascending = False)
# Regular/loyal customer:
df.groupby(['Customer ID','Customer Name']).count()['Row ID'].sort_values(ascending=False)
# Creating Order_Month Column:
import datetime as dt
df['Order Date']
df['Order_Month']=df['Order Date'].dt.month
df
df['Order Date'].dt.strftime('%B')
# creating function:

def season(x):
    # spring: March to May
    if x>=3 and x<=5:
        a = 'Spring'
    
    # summer: June to August
    elif x>=6 and x<=8:
        a = 'Summer'
    
    # fall(autmn): September to November
    elif x>=9 and x<=11:
        a = 'Fall'
    
    # winter: December to February:
    else:
        a = 'Winter'
    return a
# creating season column:

df['Season'] = df['Order_Month'].apply(season)
df
pd.crosstab(df['Season'], df['Category']).plot(kind='bar',stacked=True)
plt.xlabel('Season')
plt.ylabel('Quantity')
plt.show()
# creating Year column:
df['Year'] = df['Order Date'].dt.year
df.head()
df.groupby(['Year']).max()['Profit']
df.groupby(['Year'],)['State','Profit']
# year and state wise maximum profit
df.sort_values('Profit', ascending=False).groupby('Year')['State','Profit'].first()
df.sort_values('Sales', ascending=False).groupby('Year')['State','Sales'].first()
