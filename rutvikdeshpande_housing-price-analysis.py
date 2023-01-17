import pandas as pd

import numpy as np

import datetime as dt

import matplotlib.pyplot as plt

import seaborn as sns
df = pd.read_csv(r"../input/melbourne-housing-market/Melbourne_housing_FULL.csv")
# Data Types

df.dtypes
df.isnull().sum()
df_new = df.fillna(method = "ffill")

df_new.isnull().sum()
df_new = df_new.dropna()

df_new.isnull().sum()
df_new['year'] = pd.DatetimeIndex(df_new['Date']).year
plt.style.use("fivethirtyeight")
## Region Wise Number of Houses

region = df_new['Regionname'].value_counts()



region.plot.bar(figsize =(20, 20), title = "Number of Houses per Region")
## Council Wise Number of Houses

CouncilArea = df_new['CouncilArea'].value_counts()



CouncilArea.plot.bar(figsize =(20, 20), title = "Number of Houses per Governing Council of Area")
## Type Wise Number of Houses

types = df_new['Type'].value_counts()



types.plot.bar(figsize =(20, 20), title = "Number of Houses per Region")
## Total Deals by Real estate agent

REA = df_new['SellerG'].value_counts()



REA.head(n=10).plot.bar(figsize =(20, 20), title = "Top Real Estate Agents\n(Number of Deals Done)")
# lets make a new column which has price per square meter area

## Price per Sq . meter of suburb



price = pd.DataFrame(df_new)



price['P/sqmtr'] = price['Price'] // price['BuildingArea']





price.replace([np.inf, -np.inf], np.nan, inplace=True) 

  

# Dropping all the rows with nan values 

price.dropna(inplace=True) 
sp = pd.DataFrame(price[['Suburb', 'P/sqmtr']].groupby(['Suburb']).agg(['mean']))



sp.columns=['Mean Price per Sq. Meter (Australian Dollars)']

sp = sp.nlargest(10, ['Mean Price per Sq. Meter (Australian Dollars)']) 

sp.plot.bar(figsize = (15,10), title = "Top 10 Suburbs with Highest Average Price / Sq.Meter")
sp1 = pd.DataFrame(price[['Suburb', 'P/sqmtr']].groupby(['Suburb']).agg(['mean']))



sp1.columns=['Mean Price per Sq. Meter (Australian Dollars)']

sp1 = sp1.nsmallest(10, ['Mean Price per Sq. Meter (Australian Dollars)']) 

sp1.plot.bar(figsize = (15,10), title = "Top 10 Suburbs with Lowest Average Price / Sq.Meter")
## Number of Rooms vs Price

plt.figure(figsize=(15,10))

sns.regplot(x = 'Rooms', y = 'Price', data = price)
## Number of Bedrooms vs Price

plt.figure(figsize=(15,10))

sns.regplot(x = 'Bedroom2', y = 'Price', data = price)
## "Landsize (mtrs) vs Price

plt.figure(figsize=(15,10))

sns.regplot(x = 'Landsize', y = 'Price', data = price)
## Built Area (mtrs) vs Price

plt.figure(figsize=(15,10))

sns.regplot(x = 'BuildingArea', y = 'Price', data = price)
## Distance of House from Central Business District vs Price

plt.figure(figsize=(15,10))

sns.regplot(x = 'Distance', y = 'Price', data = price)
## Mean Price of regions

sp2 = pd.DataFrame(price[['Regionname', 'P/sqmtr']].groupby(['Regionname']).agg(['mean']))

sp2.plot.bar(figsize =(15, 10), title = "Average Price per Sq. Foot of each Region")
## Year built vs price

sp3 = pd.DataFrame(price[['YearBuilt', 'Price']].groupby(['YearBuilt']).agg(['mean']))

sp3.plot.line(figsize =(15, 10), title = "Average Price of a House according to the Year Built")

plt.xlim([1800, 2018])
## Price and Car

sp4 = pd.DataFrame(price[['Car', 'Price']].groupby(['Car']).agg(['mean']))

sp4.plot.bar(figsize =(15, 10), title = "Average Price according to number of Cars in the House")
## Richest Seller/ real estate agent

# Real estate agents with highest total sale 

ra = pd.DataFrame(price[['SellerG', 'Price']].groupby(['SellerG']).agg(['sum']))



ra.columns=['Total_Sale']

ra = ra.nlargest(10, ['Total_Sale']) 

ra.plot.bar(figsize = (15,10), title = "Top 10 Real Estate Agents with highest total sale")
## Boxplot Comparison of Prices and Method of Sale

price.boxplot('Price','Method', figsize = (15,10), color = "blue")
plt.figure(figsize = (15,10))

sns.heatmap(df_new.corr(), annot = True)