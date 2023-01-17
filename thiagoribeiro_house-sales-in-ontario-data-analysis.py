#Importing Data Analysis Libs

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

import warnings

warnings.filterwarnings('ignore')
#Getting Data

df = pd.read_csv('../input/properties.csv')
#Checking Dataset

df.head(5)
df.columns
#Changing some Column Names

df.rename(columns={'Unnamed: 0': 'ID', 'Price ($)': 'Price'}, inplace=True)

df.columns
df.shape
df.describe()
#Eliminating houses with invalid lat and lng

df2 = df[(df['lat'] != -999 ) & (df['lng'] != -999)]

df2.shape
#Eliminating houses with price <= 100000 and >= 10000000

df3 = df2[(df2['Price'] >= 100000) & (df2['Price'] <= 10000000)]

df3.shape
df3.describe()
#Creating a TOP 20 Rank by area

qtHouses = pd.DataFrame(df3['AreaName'].value_counts())

qtHouses.sort_values(by='AreaName')

qtHouses[0:20].plot(kind='bar', title='Top 20 Areas by Number of Houses on Sale')
#Boxplot 

topAreasNames = list(qtHouses[0:20].index)

area = df3[df3['AreaName'].isin(topAreasNames)]

#print(topAreasNames)

box_ax = area.boxplot(column='Price', by='AreaName', rot=90, grid=True)

box_ax.set_ylim(-1e5, 4e6)



plt.show()
#Mean Price by Area(Top 20)

df3_dropped = df3.drop(['ID','Address','lat','lng'], axis=1)

meanPrices = df3_dropped.groupby(['AreaName']).mean().sort_values(by='Price', axis=0, ascending=False)

topMeanPrices = meanPrices[0:20] 

topMeanPrices.plot(kind='bar', title='Top 20 Mean Prices of Houses by Area')