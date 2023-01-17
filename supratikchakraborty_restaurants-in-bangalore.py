import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns

import re

%matplotlib inline

import os

print(os.listdir("../input"))
#Read the zomato.csv file

train=pd.read_csv("../input/zomato.csv")
#get a look at the data

train.head(3)
#Info about the data

train.info()
#Coping the data to another variable

#Dropping unnecessary columns 

data=train.copy()

data.drop(columns=['url','address','phone','reviews_list'],inplace=True)
data['rate']=data['rate'].str.slice(stop=3)



data['rate']=data['rate'].replace('NEW','0.0')

data['rate']=data['rate'].replace('-','0.0')

data['rate'].fillna('0.0',inplace=True)

data['approx_cost(for two people)'].fillna('0.0',inplace=True)

data['approx_cost(for two people)'].replace(regex=True,inplace=True,to_replace=r'\D',value=r'')

data['rate']=pd.to_numeric(data['rate'],errors='coerce')

data['approx_cost(for two people)']=pd.to_numeric(data['approx_cost(for two people)'],errors='coerce')

#Checking Location data

data[data['location'].isnull()]



data = data[data['location'].notnull()]
data = data[data['location'].notnull()]
print(' Maximum votes received is {} \n Average votes received is {} \n Minimum votes received is {}'.format(data['votes'].max(),data['votes'].mean(),data['votes'].min()))
# Most Rated Restaurent in Bangalore

print('Most Rated Restaurent in Bangalore')

data[data['votes']==16832]
# Other top 5 Restaurents in Bangalore

print('Other Good Restaurents in Bangalore')

data[(data['votes']>1000) & (data['rate']>4.0)].groupby(by=data['name']).mean().head()
#Costliest Restaurent in Bangalore

print('Costliest Restaurent in Bangalore')

data[data['approx_cost(for two people)']==data['approx_cost(for two people)'].max()]
#Cheapest Restaurent in Bangalore

print('Cheapest Restaurent in Bangalore')

data[data['approx_cost(for two people)']==data['approx_cost(for two people)'].min()].head(5)
#Best Restaurents have good ratings, votes and also a affordable price

print("Best Restaurents in Bangalore :")

data[(data['rate']>4.5) & (data['votes']>2000) & (data['approx_cost(for two people)']<1000)].groupby(by=data['name']).mean()
fig=plt.figure(figsize = (10,5))

ax=fig.add_axes([0.1,0.1,1,1])

sns.countplot(x=data['listed_in(type)'],hue=data['online_order'],ax=ax)

ax.set_title("Type of Restaurent in Bangalore with respect to providing online order service")
fig=plt.figure(figsize = (10,5))

ax=fig.add_axes([0.1,0.1,1,1])

sns.countplot(x=data['listed_in(type)'],hue=data['book_table'],ax=ax)

ax.set_title("Type of Restaurent in Bangalore with respect to providing book table service")
data_loc=data.groupby(by=[data['listed_in(city)'],data['listed_in(type)']]).count()

data_loc.drop(['name','online_order', 'book_table', 'rate', 'votes', 'location','rest_type', 'dish_liked', 'cuisines','menu_item'], axis=1,inplace=True)

print('Below Table gives Approximate Cost for Two People in various places on various types of food in bangalore')

data_loc.unstack()
print('Top 15 Franchisee having most number of outlets in Bangalore : \n{}'.format(data['name'].value_counts().head(15)))
plt.figure(figsize=(15,5))

ax=sns.countplot(x=data['listed_in(city)'], data=data)

ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")

ax.set_title('Distribution Of Restaurents throughout Bangalore')

plt.tight_layout()

plt.show()
fig,ax=plt.subplots(nrows=2,ncols=2,figsize=(25,10))



ax0=sns.scatterplot(x=data['rate'],y=data['approx_cost(for two people)'],hue=data['online_order'],data=data,ax=ax[0,0])

ax1=sns.scatterplot(x=data['rate'],y=data['approx_cost(for two people)'],hue=data['book_table'],data=data,ax=ax[0,1])

ax2=sns.distplot(a=data['approx_cost(for two people)'],ax=ax[1,0])

ax3=sns.barplot(x=data['rate'],y=data['votes'],data=data)



ax0.set_xticklabels(labels=[-1,0,1,2,3,4,5])

ax1.set_xticklabels(labels=[-1,0,1,2,3,4,5])

ax3.set_xticklabels(ax3.get_xticklabels())



ax0.set_title('Approx cost w.r.t rate and online_order')

ax1.set_title('Approx cost w.r.t rate and book_table')

ax2.set_title('Distribution of Approx cost in the whole data set')

ax3.set_title('Relation b/w rate and votes in this dataset')

plt.show()