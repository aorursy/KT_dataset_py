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
import numpy as np

import pandas as pd

import os

import sklearn

import matplotlib as mpl

import matplotlib.pyplot as plt

import seaborn as sns

import datetime



# Reading the file



data=pd.read_csv("/kaggle/input/customer-sales-data/Copy of transaction_data.csv")

data.head(5)
data.describe()
# Extracting the date, day ,month, year from "TransactionTime" for furthur use



data_time=np.array(data['TransactionTime'])



day=[]

month=[]

date=[]

time=[]

year=[]



i=0

while i<len(data_time):

  day.append(data_time[i][0:3])

  date.append(data_time[i][8:10])

  month.append(data_time[i][4:7])

  time.append(data_time[i][11:19])

  year.append(data_time[i][-4:])

  i=i+1



date=np.array(date)

day=np.array(day)

year=np.array(year)

time=np.array(time)



data['Day']=day

data['Date']=date

data['Time']=time

data['Year']=year

data['Month']=month
# Converting the string values of month into numeric and creating a new column of date and time only for arranging the dataframe as per transation date and time

# the new column is "TransactionTime"

Transactiondate=[]

k='00'

for i in range(len(year)):

  if month[i]=='Jan':

    k='01'

  elif month[i]=='Feb':

    k='02'

  elif month[i]=='Mar':

    k='03'

  elif month[i]=='Apr':

    k='04'

  elif month[i]=='May':

    k='05'

  elif month[i]=='Jun':

    k='06'

  elif month[i]=='Jul':

    k='07'

  elif month[i]=='Aug':

    k='08'

  elif month[i]=='Sep':

    k='09'

  elif month[i]=='Oct':

    k='10'

  elif month[i]=='Nov':

    k='11'

  elif month[i]=='Dec':

    k='12'

  Transactiondate.append(str(year[i])+'-'+str(k)+'-'+str(date[i])+' '+str(time[i]))



Transactiondate=np.array(Transactiondate)

data['Transactiondate']=Transactiondate
data.head(5)
# creating new column  "Total_cost"=NumberOfItemsPurchased * CostPerItem



data7=np.asarray(data['NumberOfItemsPurchased']*data['CostPerItem']).astype(float)

data['Total_Cost']=data7

data.head(5)
#copying the datframe into another variable for further use



df_initial=pd.DataFrame(data[['UserId','TransactionId','Transactiondate','ItemCode','ItemDescription','NumberOfItemsPurchased','CostPerItem','Total_Cost','Country','Day','Time','Date','Month','Year']])

df_initial.head(2)

#removing the rows in which the UserId is not known



print('Dataframe dimensions:', df_initial.shape)

indexNames = df_initial[ df_initial['UserId'] == -1 ].index

df_initial.drop(indexNames, inplace = True)

print('Dataframe dimensions after dropping unknown values:', df_initial.shape)





#removing the rows having duplicate values



print('Duplicate entries: {}'.format(df_initial.duplicated().sum()))

df_initial.drop_duplicates(inplace = True)

print('Dataframe dimensions after removing duplicate values:', df_initial.shape)
#sale of items on difeerent days



sns.set(style="darkgrid")       

DaySale  = sns.countplot( x='Day',data =df_initial).set_title("Day Sale")
# sale of items in different months

MonthSale  = sns.countplot( x='Month',data =df_initial).set_title("Month Sale")
#counting the number of countries form where transaction has been done



temp = df_initial[['UserId', 'TransactionId', 'Country']].groupby(['UserId', 'TransactionId', 'Country']).count()

temp = temp.reset_index(drop = False)

countries = temp['Country'].value_counts()

print('No of countries: {}'.format(len(countries)))
# representing the sale from differetn countries in world map



data = dict(type='choropleth',locations = countries.index,locationmode = 'country names', z = countries,text = countries.index, colorbar = {'title':'Order nb.'},

colorscale=[[0, 'rgb(224,255,255)'],

            [0.01, 'rgb(166,206,227)'], [0.02, 'rgb(31,120,180)'],

            [0.03, 'rgb(178,223,138)'], [0.05, 'rgb(51,160,44)'],

            [0.10, 'rgb(251,154,153)'], [0.20, 'rgb(255,255,0)'],

            [1, 'rgb(227,26,30)']],    

reversescale = False)



layout = dict(title='Number of orders per country',

geo = dict(showframe = True, projection={'type':'mercator'}))



choromap = go.Figure(data = [data], layout = layout)



iplot(choromap, validate=False)
# printing the quantity of unique products sold , the number of transactions done , the number of customers



pd.DataFrame([{'products': len(df_initial['ItemCode'].value_counts()),    

               'transactions': len(df_initial['TransactionId'].value_counts()),

               'customers': len(df_initial['UserId'].value_counts()),  

              }], columns = ['products', 'transactions', 'customers'], index = ['quantity'])
#printing the data of numer of items purchased by a user in a every unique transaction



temp = df_initial.groupby(by=['UserId', 'TransactionId'], as_index=False)['Transactiondate'].count()

nb_products_per_basket = temp.rename(columns = {'Transactiondate':'NumberOfItemsPurchased'})

nb_products_per_basket[:10].sort_values('UserId')


df_initial.sort_values('UserId')[:5]
#printing the basket price at end of every transaction

#basket price contains the cost of every items purchased the number of times * Coset per item



df_initial['Transactiondate'] = pd.to_datetime(df_initial['Transactiondate'])

temp = df_initial.groupby(by=['UserId', 'TransactionId'], as_index=False)['Total_Cost'].sum()

basket_price = temp.rename(columns = {'Total_Cost':'Basket_Price'})



df_initial['Transactiondate_int'] = df_initial['Transactiondate'].astype('int64')

temp = df_initial.groupby(by=['UserId', 'TransactionId'], as_index=False)['Transactiondate_int'].mean()

df_initial.drop('Transactiondate_int', axis = 1, inplace = True)

basket_price.loc[:, 'Transactiondate'] = pd.to_datetime(temp['Transactiondate_int'])

basket_price = basket_price[basket_price['Basket_Price'] > 0]

basket_price.sort_values('UserId')[:6]


df_products = pd.DataFrame(df_initial['ItemDescription'].unique())

# count contains the value of item and its frequency value in whole dataset

count = df_initial['ItemDescription'].value_counts()



#most_buyed has items buyed more than 1000 times

most_buyed=count[count > 1000] 

print(most_buyed)
# displaying every user first purchase date and last purchase date with the number of transactions done and the value of minimum,maximum,mean and total transaction value



basket_price['Transactiondate'] = basket_price['Transactiondate'].astype('int64')

basket_price['Transactiondate'] = pd.to_datetime(basket_price['Transactiondate'])

transactions_per_user=basket_price.groupby(by=['UserId'])['Basket_Price'].agg(['count','min','max','mean','sum'])



first_registration = pd.DataFrame(basket_price.groupby(by=['UserId'])['Transactiondate'].min())

last_purchase= pd.DataFrame(basket_price.groupby(by=['UserId'])['Transactiondate'].max())



transactions_per_user['FirstPurchase'] = first_registration

transactions_per_user['LastPurchase'] = last_purchase



transactions_per_user[0:5]


# printing the the percentage of customes with only single order

n1 = transactions_per_user[transactions_per_user['count'] == 1].shape[0]

n2 = transactions_per_user.shape[0]

print("Percentage of Customers with only single purchase: {:<2}/{:<5} ({:<2.2f}%)".format(n1,n2,n1/n2*100))
#printing all the items sekected in every transaction by the user



df_initial['Transactiondate'] = pd.to_datetime(df_initial['Transactiondate'])

temp = pd.DataFrame(df_initial.groupby(by=['UserId', 'TransactionId'], as_index=False)['ItemDescription'].apply(", ".join),columns = ['Items selected in basket'])



temp
temp1.head(2)


# temp2 has value of all unique userid and all the products they haved buyed till now





temp1=pd.DataFrame(temp)

temp1=temp1.reset_index(level='UserId')

temp1.reset_index(drop=True, inplace=True)

temp1.head(5)



temp2 = pd.DataFrame(temp1.groupby(by=['UserId'], as_index=False)['Items selected in basket'].apply(", ".join),columns = ['All_items_buyed_buy_user'])

user=temp1['UserId'].unique()

temp2['UserId']=user

temp2.head(5)