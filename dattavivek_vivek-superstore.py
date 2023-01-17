# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



import matplotlib.pyplot as plt      # For base plotting



import seaborn as sns                # Easier plotting



import os



data = pd.read_csv("../input/superstore_dataset2011-2015.csv",encoding = 'unicode_escape')

#removing the inbetween spaces of columns.

data.rename(columns=lambda x: x.replace(' ',''),inplace=True)

data.shape

# Any results you write to the current directory are saved as output.
#creating a new column for daynames

data['DayName']=pd.to_datetime(data['OrderDate']).dt.day_name()



#converting object to datetime columns

data['OrderDate']=pd.to_datetime(data['OrderDate'],dayfirst=True)

data['ShipDate']=pd.to_datetime(data['ShipDate'],dayfirst=True)



#creating a new column to know the shipment time in days

data['shipmentTime']=(data['ShipDate']-data['OrderDate']).dt.days

data[['DayName','shipmentTime']].head(10)
#what are the busiest days in a week?specific to any year of a country,region,Market....

dataInd13=data.loc[(data['OrderDate'].dt.year==2013) & (data['Country']=='India')]

dataAus13=data.loc[(data['OrderDate'].dt.year==2013) & (data['Country']=='Australia')]

dataChina13=data.loc[(data['OrderDate'].dt.year==2013) & (data['Country']=='China')]



dataAus13.groupby('DayName').size().reset_index(name='count').sort_values('count',ascending=False)

dataChina13.groupby('DayName').size().reset_index(name='count').sort_values('count',ascending=False)

dataInd13.groupby('DayName').size().reset_index(name='count').sort_values('count',ascending=False)

#total orders distribution with respect to the days.

totaldf=data.groupby('DayName').size().reset_index(name='count').sort_values('count', ascending=False).rename(columns={'count':'TotalOrders'})

sns.jointplot("DayName",

              "TotalOrders",

              totaldf,

              kind='scatter'  # kind : { “scatter” | “reg” | “resid” | “kde” | “hex” }

              )
#distribution of global sales across the years

Globalsales=data.groupby(data['OrderDate'].dt.year).size().reset_index(name='count').sort_values('count', ascending=False).rename(columns={'OrderDate':'Year','count':'Sales'})

Globalsales

sns.barplot(x = "Year",     

            y= "Sales",    

            data=Globalsales

            )
#distribution of sales in market and  across the years

Marketsales=data.groupby([data['OrderDate'].dt.year,'Market']).size().reset_index(name='count').sort_values('count', ascending=False).rename(columns={'OrderDate':'Year','count':'Sales'})

Marketsales

sns.barplot(x = "Year",  

            y= "Sales", 

            hue="Market",

            data=Marketsales

            )
#distribution of sales in India across the years

dataInd=data.loc[(data['Country']=='India')].groupby(data['OrderDate'].dt.year).size().reset_index(name='count').sort_values('count', ascending=False).rename(columns={'OrderDate':'Year','count':'Sales'})



sns.jointplot("Year",

              "Sales",

              dataInd,

              kind='scatter'  # kind : { “scatter” | “reg” | “resid” | “kde” | “hex” }

              )
#To know more about the shipment time like 'min','max','mean' based on low,medium and high priority orders for different countries.

IndMean=dataInd13.groupby('OrderPriority').agg({'shipmentTime':['min','max','count','mean']}).rename(columns={'mean': 'IndShipTime'}).reset_index()

AusMean=dataAus13.groupby('OrderPriority').agg({'shipmentTime':['min','max','count','mean']}).rename(columns={'mean': 'AusShipTime'}).reset_index()

IndMean
#distribution of average shipping time across the global markets

avgShipping=data[['Market','OrderPriority','shipmentTime']].groupby(['OrderPriority','Market']).mean().reset_index().sort_values('shipmentTime', ascending=False)



sns.barplot(x = "OrderPriority",     # Data is groupedby this variable

            y= "shipmentTime",    # Aggregated by this variable

                               # Continuous variable. Bar-ht,

                               

            hue= "Market",     # Distribution is gender-wise

            data=avgShipping

            )
#number of orders placed for different discount ranges based on category

data.loc[data['Discount']>=0.40].groupby('Category').size().reset_index(name='count').sort_values('count', ascending=False)



lessDiscountdf=data.loc[data['Discount']<0.40].groupby('Category').size().reset_index(name='count').sort_values('count', ascending=False)

lessDiscountdf
#find out the avg shipping cost distribution for top 20 different countries

data.groupby('Country').agg({'ShippingCost':'mean'}).sort_values('ShippingCost', ascending=False).head(20)

#Who are the top-20 most profitable customers.

data20=data.sort_values('Profit',ascending=False).head(20)[['CustomerID','CustomerName']]

data20
 #What is the distribution of our customer segment

sns.countplot("Segment", data = data)
# Which customers have visited this store just once

df1 = data.groupby('CustomerID').apply(lambda x: pd.Series(dict(onevisit=x.shape[0]))).reset_index()

df1.loc[df1.onevisit == 1, ['CustomerID', 'onevisit']] 
#6 What is the distribution of orders for Market and region wise?

sns.countplot('Market',hue='Region',data= data)