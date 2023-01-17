import pandas as pd 

import numpy as np  

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



import warnings as war

war.filterwarnings('ignore')



import os

print(os.listdir("../input/zomato-bangalore-restaurants/"))
data=pd.read_csv('../input/zomato-bangalore-restaurants/zomato.csv')

data.head()
data.shape
print("Percentage of null values in df-")

(data.isnull().sum()*100/data.index.size).round(2)
data['rate']=data['rate'].replace('NEW',np.nan)

data['rate'].unique()
data.dropna(how='any',inplace=True)
data['rate']=data['rate'].apply(lambda x: x.split('/')[0])
data.head(2)
del data['url']

del data['address']

del data['location']

del data['phone']

del data['menu_item']
data.head()
data.rename(columns={'listed_in(type)': 'Restaurant_type','listed_in(city)':'location','approx_cost(for two people)':'average_cost'},inplace=True)
data.head()
data['average_cost']=data['average_cost'].apply(lambda x: x.replace(',',''))

data['average_cost']=data['average_cost'].astype(int)
data['average_cost'].dtype
data.head()
print(data['Restaurant_type'].unique())

data['rate']=data['rate'].astype('float')
sns.countplot(data['online_order'])
data['online_order'].value_counts()
plt.figure(figsize=(15,7))

g=sns.countplot(data['location'],palette='Set1',hue=data['online_order'])

g.set_xticklabels(g.get_xticklabels(), rotation=90, ha='right')

g
data[data['online_order']=='Yes']['online_order'].count()
New_df=data[data['online_order']=='Yes'].sort_values(by='votes',ascending=False)

#New_df.groupby('name').first()
data['Restaurant_type'].value_counts()
data[data['online_order']=='Yes']['Restaurant_type'].value_counts()
print('No of Restaurant doesnot accept Online Orders are: ',10575-9179)
data[data['Restaurant_type']=='Delivery']['online_order'].value_counts()
Online_No=data[(data['Restaurant_type']=='Delivery') & (data['online_order']=='No')]

Online_No.groupby('name').first()
Online_No['name'].value_counts()
Online_No[Online_No['book_table']=='No']['name'].value_counts()
Top1000_byrate=data.sort_values(by='rate',ascending=False).head(1000)

print('Top 20 Restaurants with the highest rates')

Top1000_byrate[['name','rate','votes','rest_type','average_cost','location']].head(20)
Economical_Restaurents=Top1000_byrate.sort_values(by='average_cost')

Economical_Restaurents.head()
Economical_Restaurents['cuisines'].value_counts()
Economical_Restaurents[Economical_Restaurents['cuisines']=='Desserts']
Economical_Restaurents[Economical_Restaurents['cuisines']=='Desserts']['name'].unique()
Economical_Restaurents[Economical_Restaurents['cuisines']=='Desserts'].sort_values(by='votes',ascending=False).head(2)
Top1000_byprice=data.sort_values(by='average_cost',ascending=False).head(1000)
print ('Top 20 most expensive Restaurent')

Top1000_byprice[['name','rate','votes','rest_type','average_cost','location']].head(20)
Best_and_expensive=Top1000_byprice.sort_values(by='rate',ascending=False)

Best_and_expensive.head()