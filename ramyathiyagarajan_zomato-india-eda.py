# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns

import os

os.listdir('../input')
df=pd.read_csv('../input/zomato-restaurants-in-india/zomato_restaurants_in_India.csv')
df.head()
l=['res_id','url','address','city_id','latitude','longitude','zipcode','country_id','timings']

df=df.drop(l,axis=1)
df=df.drop('currency',axis=1)
delhi=df[df['city']=='New Delhi']

bangalore=df[df['city']=='Bangalore']

chennai=df[df['city']=='Chennai']

mumbai=df[df['city']=='Mumbai']

delhi_greaterthan4=delhi[delhi['aggregate_rating']>4.0]

bang_greaterthan4=bangalore[bangalore['aggregate_rating']>4.0]

chennai_greaterthan4=chennai[chennai['aggregate_rating']>4.0]

mum_greaterthan4=mumbai[mumbai['aggregate_rating']>4.0]
rating=pd.concat([delhi_greaterthan4,bang_greaterthan4])

rating=pd.concat([rating,mum_greaterthan4,chennai_greaterthan4])
sns.countplot(x='city',data=rating)
rating.establishment.replace("['Quick Bites']",'quick bites',inplace=True)

rating.establishment.replace("['Dessert Parlour']",'Dessert Parlour',inplace=True)

rating.establishment.replace("['Bakery']",'bakery',inplace=True)

rating.establishment.replace("['Café']",'cafe',inplace=True)

rating.establishment.replace("['Sweet Shop']",'sweet shop',inplace=True)

rating.establishment.replace("['Mess']",'mess',inplace=True)

rating.establishment.replace("['Beverage Shop']",'beverage shop',inplace=True)

rating.establishment.replace("['Food Truck']",'food truck',inplace=True)

rating.establishment.replace("['Casual Dining']",'Casual Dining',inplace=True)

rating.establishment.replace("['Fine Dining']",'Fine Dining',inplace=True)

rating.establishment.replace("['Lounge']",'Lounge',inplace=True)

rating.establishment.replace("['Bar']",'Bar',inplace=True)

rating.establishment.replace('[]','Not_mentioned',inplace=True)

rating.establishment.replace("['Food Truck']",'food truck',inplace=True)

rating.establishment.replace("['Kiosk']",'Kiosk',inplace=True)

rating.establishment.replace("['Club']",'Club',inplace=True)

rating.establishment.replace("['Paan Shop']",'Paan',inplace=True)

rating.establishment.replace("['Microbrewery']",'Microbrewery',inplace=True)

rating.establishment.replace("Food Truck']",'food truck',inplace=True)

rating.establishment.replace("['Pub']",'Bar',inplace=True)

rating.establishment.replace("['Dhaba']",'Dhaba',inplace=True)

rating.establishment.replace("['Cocktail Bar']",'Bar',inplace=True)

rating.establishment.replace("['Food Court']",'Street_food',inplace=True)

rating.establishment.replace("Kiosk",'Street_food',inplace=True)

rating.establishment.replace("food truck",'Street_food',inplace=True)
rating=pd.get_dummies(data=rating,columns=['establishment'])

sns.barplot(x=rating['city'],y=rating['establishment_Bar'])
sns.barplot(x=rating['city'],y=rating['establishment_Casual Dining'])
sns.barplot(x=rating['city'],y=rating['establishment_Club'])
sns.barplot(x=rating['city'],y=rating['establishment_Dessert Parlour'])
sns.barplot(x=rating['city'],y=rating['establishment_Fine Dining'])
sns.barplot(x=rating['city'],y=rating['establishment_Microbrewery'])
sns.barplot(x=rating['city'],y=rating['establishment_Dhaba'])
sns.barplot(x=rating['city'],y=rating['establishment_Paan'])
sns.barplot(x=rating['city'],y=rating['establishment_Lounge'])
sns.barplot(x=rating['city'],y=rating['establishment_Club'])
sns.barplot(x=rating['city'],y=rating['establishment_Street_food'])
sns.barplot(x=rating['city'],y=rating['establishment_bakery'])
sns.barplot(x=rating['city'],y=rating['establishment_cafe'])
sns.barplot(x=rating['city'],y=rating['establishment_mess'])
sns.barplot(x=rating['city'],y=rating['establishment_quick bites'])
sns.barplot(x=rating['city'],y=rating['establishment_sweet shop'])
rating.groupby('city')['average_cost_for_two'].median()
sns.countplot(rating['delivery'])
sns.barplot(x=rating['city'],y=rating['price_range'])