import pandas as pd

import numpy as np

import os

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

pd.options.display.max_columns = 300

# 2.1 Read file

data = pd.read_csv("../input/superstore_dataset2011-2015.csv", encoding = "ISO-8859-1")

data.shape
data.columns 
# function to call data[name_column].describe()

def describe_col(name_column):

    return data[name_column].describe()
# Check Is Null

data.isnull().sum()
# 4.4 Head 5 Rows (Default)

data.head()
describe_col('Order ID')
describe_col('Customer ID')
data['Ship Date'] = pd.to_datetime(data['Ship Date'])



describe_col('Ship Date')
# Shipping Mode wise sales



plt.figure(figsize=(16,8))

data['Ship Mode'].value_counts().plot.bar()

plt.title('Shipping Mode Wise Sales')

plt.ylabel('Sales')

plt.xlabel('Ship Modes')

plt.show()



# Standard Class shipping method is the highest which is 12 times higher than same day. 

# if they decrease the price of the other shipping method that could increase the customer satisfaction

data['Order Date'] = pd.to_datetime(data['Order Date'])



describe_col('Order Date')
describe_col('Ship Mode')
data['Ship Mode'].unique()
describe_col('Customer Name')

data['Customer Name'].unique()
describe_col('City')

data['City'].unique()
# Top 20 Cities in Sales

plt.figure(figsize=(16,8))

top20city = data.groupby('City')['Row ID'].count().sort_values(ascending=False)

top20city = top20city [:15]

top20city.plot(kind='bar')

plt.title('Top 15 Cities in Sales')

plt.ylabel('Count')

plt.xlabel('Cities')

plt.show()





# New York City tops all the Cities in Sales followed by LA

describe_col('Segment')

data['Segment'].unique()
# Segment

plt.figure(figsize=(16,8))

data['Segment'].value_counts().plot.bar()



plt.title('Segment Wise Sales')

plt.ylabel('Count')

plt.xlabel('Segments')

plt.show()

# Consumers are the biggest buyers then corportes and then Home office

# Company should try to bring more schemes for the consumers

# to improve the corporate sales, they can bring sorporate level schemes
describe_col('State')

data['State'].unique()
# Top 10 Stateswise Sales

plt.figure(figsize=(16,8))

top20states = data.groupby('State')['Row ID'].count().sort_values(ascending=False)

top20states = top20states [:10]

top20states.plot(kind='bar')

plt.title('Top 10 States by Sales')

plt.ylabel('Count')

plt.xlabel('States')

plt.show()







# California tops all the States in Sales followed by England then New York
describe_col('Country')

data['Country'].unique()
# Top 20 Countries in sales

plt.figure(figsize=(16,8))

top20countries = data.groupby('Country')['Row ID'].count().sort_values(ascending=False)

top20countries = top20countries [:20]

top20countries.plot(kind='bar')

plt.title('Top 20 Countries by Sales')

plt.ylabel('Count')

plt.xlabel('Countries')

plt.show()





# US tops the sales by 3 times than all the Countries

describe_col('Market')

data['Market'].unique()
# Market Vs Sales

plt.figure(figsize=(16,8))

data['Market'].value_counts().plot.bar()

plt.title('Market Wise Sales')

plt.ylabel('Count')

plt.xlabel('Region')

plt.show()

# 4 regions are approximately at the top APAC, EU, US and LATM
describe_col('Region')

data['Region'].unique()
describe_col('Product ID')

data['Product ID'].unique()
describe_col('Category')

data['Category'].unique()
# Category wise

plt.figure(figsize=(16,8))

data['Category'].value_counts().plot.bar()

plt.title('Category Wise Sales')

plt.ylabel('Sales')

plt.xlabel('Categories')

plt.show()





# Office Supplies tops all the Sales in Categories



describe_col('Sub-Category')

data['Sub-Category'].unique()
# Sub-Category wise sales



plt.figure(figsize=(16,8))

data['Sub-Category'].value_counts().plot.bar()

plt.title('Sub-Category Wise Sales')

plt.ylabel('Sales')

plt.xlabel('Sub Categories')

plt.show()



# Binders are in big demand - 
describe_col('Product Name')

data['Product Name'].unique()
# Top 20 Product in Sales



plt.figure(figsize=(16,8))

top20pid = data.groupby('Product Name')['Row ID'].count().sort_values(ascending=False)

top20pid = top20pid [:20]

top20pid.plot(kind='bar')

plt.title('Top 20 Products in Sales')

plt.ylabel('Count')

plt.xlabel('Product')

plt.show()



# "Staples" seems to top all the Products in Sales by big margin
describe_col('Sales')
describe_col('Discount')
describe_col('Profit')
# Top 5 customers who earned profits





plt.figure(figsize=(16,8))

profitable = data.sort_values('Profit', ascending=False)

top20 = profitable.head(5)

top20[['Customer Name', 'Profit']]

sns.barplot(x = "Customer Name", y= "Profit", data=top20)  



# Tamara Chand tops the profitable chart


data['Order Priority'].unique()
# Order Priority



plt.figure(figsize=(16,8))

data['Order Priority'].value_counts().plot.bar()

plt.title('Order Priority Wise Sales')

plt.ylabel('Sales')

plt.xlabel('Order Priorities')

plt.show()



# Mostly the Orders are placed with medium priority, critical are very less




# Relationship of Order Priority and Profitability



plt.figure(figsize=(16,8))

sns.boxplot("Order Priority","Profit",data= data)

plt.title('Order Priority and Profitability')

plt.show()



# Profits are higher for the Medium priority




# Top 20 oldest Customers



data['Order Date'] = pd.to_datetime(data['Order Date'])      

top20Cust= data.sort_values(['Order Date'], ascending=False).head(20)

top20Cust.loc[:,['Customer Name']]

# Sales by product Category, Sub-category

plt.figure(figsize=(16,8))

sale_category = data.groupby(["Category","Sub-Category"])['Quantity'].aggregate(np.sum).reset_index().sort_values('Quantity',ascending = False)

sns.barplot(x = "Category", hue="Sub-Category", y= "Quantity", data=sale_category)

plt.show()



# Binders in Office Supplies tops the list.
# Customer Segment - Market wise



plt.figure(figsize=(24,15))

sns.catplot(x="Segment", col="Market", data=data, kind="count")

plt.show()



# Distribution of  Customers by Country & State - top 20

plt.figure(figsize=(16,8))



CusCountry = pd.DataFrame({'Count' : data.groupby(["Country","State"]).size()}).reset_index().sort_values('Count',ascending = False).head(20)

sns.barplot(x = "Country", y= "Count", hue="State", data = CusCountry.sort_values('Country'))



plt.show()





## US has the largest number of customers 

## UK has the next largest population of Customers
# Customers with fewer visits



Visit=data.groupby('Customer ID').apply(lambda x: pd.Series(dict(visit_count=x.shape[0])))

Visit.loc[(Visit.visit_count < 5)]



# 46Customers had very few visits....less than 5.