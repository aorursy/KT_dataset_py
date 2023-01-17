import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import plotly.express as px

import plotly.graph_objects as go

from plotly.subplots import make_subplots

!pip install pywaffle --quiet

from pywaffle import Waffle

from wordcloud import WordCloud
df= pd.read_csv("../input/nyc-property-sales/nyc-rolling-sales.csv")
df
df.iloc[:8,:10]
df.iloc[:8,10:20]
df.info()
#SALE PRICE is object but should be numeric

df['SALE PRICE'] = pd.to_numeric(df['SALE PRICE'], errors='coerce')



#LAND and GROSS SQUARE FEET is object but should be numeric

df['LAND SQUARE FEET'] = pd.to_numeric(df['LAND SQUARE FEET'], errors='coerce')

df['GROSS SQUARE FEET']= pd.to_numeric(df['GROSS SQUARE FEET'], errors='coerce')



#SALE DATE is object but should be datetime

df['SALE DATE'] = pd.to_datetime(df['SALE DATE'], errors='coerce')



#Both TAX CLASS attributes should be categorical

df['TAX CLASS AT TIME OF SALE'] = df['TAX CLASS AT TIME OF SALE'].astype('category')

df['TAX CLASS AT PRESENT'] = df['TAX CLASS AT PRESENT'].astype('category')
#Set the size of the plot

plt.figure(figsize=(15,6))



# Plot the data and configure the settings

sns.boxplot(x='SALE PRICE', data=df)

plt.ticklabel_format(style='plain', axis='x')

plt.title('Boxplot of SALE PRICE in USD')

plt.show()
# Remove observations with missing SALE PRICE

df = df[df['SALE PRICE'].notnull()]

len(df)
# Removes all NULL values

df = df[df['LAND SQUARE FEET'].notnull()] 

df = df[df['GROSS SQUARE FEET'].notnull()] 
#Set the size of the plot

plt.figure(figsize=(15,6))



#Get the data and format it

x = df[['SALE PRICE']].sort_values(by='SALE PRICE').reset_index()

x['PROPERTY PROPORTION'] = 1

x['PROPERTY PROPORTION'] = x['PROPERTY PROPORTION'].cumsum()

x['PROPERTY PROPORTION'] = 100* x['PROPERTY PROPORTION'] / len(x['PROPERTY PROPORTION'])



# Plot the data and configure the settings

plt.plot(x['PROPERTY PROPORTION'],x['SALE PRICE'], linestyle='None', marker='o')

plt.title('Cumulative Distribution of Properties according to Price')

plt.xlabel('Percentage of Properties in ascending order of Price')

plt.ylabel('Sale Price')

plt.ticklabel_format(style='plain', axis='y')

plt.show()
# Remove observations that fall outside those caps

df = df[(df['SALE PRICE'] > 100000) & (df['SALE PRICE'] < 5000000)]
#Set the size of the plot

plt.figure(figsize=(15,6))



#Get the data and format it

x = df[['SALE PRICE']].sort_values(by='SALE PRICE').reset_index()

x['PROPERTY PROPORTION'] = 1

x['PROPERTY PROPORTION'] = x['PROPERTY PROPORTION'].cumsum()

x['PROPERTY PROPORTION'] = 100* x['PROPERTY PROPORTION'] / len(x['PROPERTY PROPORTION'])



# Plot the data and configure the settings

plt.plot(x['PROPERTY PROPORTION'],x['SALE PRICE'], linestyle='None', marker='o')

plt.title('Cumulative Distribution of Properties according to Price')

plt.xlabel('Percentage of Properties in ascending order of Price')

plt.ylabel('Sale Price')

plt.ticklabel_format(style='plain', axis='y')

plt.show()
#Set the size of the plot

plt.figure(figsize=(15,6))



# Plot the data and configure the settings

sns.boxplot(x='SALE PRICE', data=df)

plt.ticklabel_format(style='plain', axis='x')

plt.title('Boxplot of SALE PRICE in USD')

plt.show()
#Set the size of the plot

plt.figure(figsize=(15,6))



# Plot the data and configure the settings

sns.distplot(df['SALE PRICE'])

plt.title('Histogram of SALE PRICE in USD')

plt.ylabel('Normed Frequency')

plt.show()
sales=np.log(df['SALE PRICE'])

print(sales.skew())

sns.distplot(sales)
plt.figure(figsize=(10,6))

sns.regplot(x='GROSS SQUARE FEET', y='SALE PRICE', data=df, fit_reg=False, scatter_kws={'alpha':0.3})

plt.title('Gross Square Feet vs Sale Price')

plt.show()
plt.figure(figsize=(10,6))

sns.regplot(x='LAND SQUARE FEET', y='SALE PRICE', data=df, fit_reg=False, scatter_kws={'alpha':0.3})

plt.title('Land Square Feet vs Sale Price')

plt.show()
# Keeps properties with fewer than 20,000 Square Feet, which is about 2,000 Square Metres

df = df[df['GROSS SQUARE FEET'] < 20000]

df = df[df['LAND SQUARE FEET'] < 20000]
plt.figure(figsize=(10,6))

sns.regplot(x='GROSS SQUARE FEET', y='SALE PRICE', data=df, fit_reg=False, scatter_kws={'alpha':0.3})

plt.title('Gross Square Feet vs Sale Price')

plt.show()
plt.figure(figsize=(10,6))

sns.regplot(x='LAND SQUARE FEET', y='SALE PRICE', data=df, fit_reg=False, scatter_kws={'alpha':0.3})

plt.title('Land Square Feet vs Sale Price')

plt.show()
#Dropping column as it is empty

del df['EASE-MENT']

#Dropping as it looks like an iterator

del df['Unnamed: 0']
#Checking for duplicated entries

sum(df.duplicated(df.columns))
#Delete the duplicates and check that it worked

df = df.drop_duplicates(df.columns, keep='last')

sum(df.duplicated(df.columns))
# Only a handful of properties with 0 total units are remaining and they will now be deleted

df = df[(df['TOTAL UNITS'] > 0) & (df['TOTAL UNITS'] < 50)]
#Remove data where commercial + residential doesn't equal total units

df = df[df['TOTAL UNITS'] == df['COMMERCIAL UNITS'] + df['RESIDENTIAL UNITS']]
df[["TOTAL UNITS", "SALE PRICE"]].groupby(['TOTAL UNITS'], as_index=False).count().sort_values(by='SALE PRICE', ascending=False)

df = df[(df['TOTAL UNITS'] > 0) & (df['TOTAL UNITS'] != 2261)] 
plt.figure(figsize=(10,6))

sns.boxplot(x='COMMERCIAL UNITS', y='SALE PRICE', data=df)

plt.title('Commercial Units vs Sale Price')

plt.show()
plt.figure(figsize=(10,6))

sns.boxplot(x='RESIDENTIAL UNITS', y='SALE PRICE', data=df)

plt.title('Residential Units vs Sale Price')

plt.show()
plt.figure(figsize=(10,6))

sns.boxplot(x='TOTAL UNITS', y='SALE PRICE', data=df)

plt.title('Total Units vs Sale Price')

plt.show()
df = df[df['YEAR BUILT'] > 0]
df.columns[df.isnull().any()]
# Compute the correlation matrix

d= df[['SALE PRICE', 'TOTAL UNITS','GROSS SQUARE FEET',  'LAND SQUARE FEET', 'RESIDENTIAL UNITS', 

         'COMMERCIAL UNITS', 'BOROUGH', 'BLOCK', 'LOT', 'ZIP CODE', 'YEAR BUILT',]]

corr = d.corr()



# Generate a mask for the upper triangle

mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True



# Set up the matplotlib figure

f, ax = plt.subplots(figsize=(11, 9))



# Generate a custom diverging colormap

cmap = sns.diverging_palette(220, 10, as_cmap=True)



# Draw the heatmap with the mask and correct aspect ratio

sns.heatmap(corr, mask=mask, 

            square=True, linewidths=.5, annot=True, cmap=cmap)

plt.yticks(rotation=0)

plt.title('Correlation Matrix of all Numerical Variables')

plt.show()
plt.figure(figsize=(10,6))

sns.boxplot(x='GROSS SQUARE FEET', data=df,showfliers=False)
plt.figure(figsize=(10,6))

sns.boxplot(x='LAND SQUARE FEET', data=df,showfliers=False)

pivot=df.pivot_table(index='TAX CLASS AT TIME OF SALE', values='SALE PRICE', aggfunc=np.median)

pivot
cat=df[["TAX CLASS AT TIME OF SALE", "SALE PRICE"]].groupby(['TAX CLASS AT TIME OF SALE'], as_index=False).mean().sort_values(by='SALE PRICE', ascending=False)

plt.figure(figsize=(20,10))

sns.barplot(x='TAX CLASS AT TIME OF SALE', y='SALE PRICE', data=cat)
cat=df[["BUILDING CLASS CATEGORY", "SALE PRICE"]].groupby(['BUILDING CLASS CATEGORY'], as_index=False).mean().sort_values(by='SALE PRICE', ascending=False)

plt.figure(figsize=(20,10))



sns.barplot(x='SALE PRICE', y='BUILDING CLASS CATEGORY', data=cat, orient = 'h')
df['SALE DATE'] = pd.to_datetime(df['SALE DATE'])

df['SALE DATE'].dtype

df['SALE DATE'] = pd.to_datetime(df['SALE DATE'])

df['YEAR SOLD'] = (df['SALE DATE']).dt.year

df['MONTH SOLD']= (df['SALE DATE']).dt.month

# del(df["SALE DATE"])
plt.subplots(figsize=(20,8))

sns.barplot(x='YEAR SOLD', y='SALE PRICE', hue='BOROUGH', data=df, palette='rainbow', ci=None)

plt.title('Sales per Borough from 2016-2017')
plt.subplots(figsize=(20,8))

sns.boxplot(x='BOROUGH', y='SALE PRICE', data=df)

plt.title('Sale Price Distribution by Borough')

plt.show()
plt.subplots(figsize=(20,8))

sns.countplot('BOROUGH',data=df,palette='Set2')

plt.title('Sales per Borough')
plt.subplots(figsize=(20,8))

sns.barplot(y='RESIDENTIAL UNITS', x='BOROUGH',data=df, palette='coolwarm', ci=None)

plt.title('Sales per borough_Residential')
plt.subplots(figsize=(20,8))

sns.barplot(y='COMMERCIAL UNITS', x='BOROUGH',data=df, palette='coolwarm', ci=None)

plt.title('Sales per borough_Commercial')
plt.figure(figsize=(20,8))

sns.barplot(x='MONTH SOLD', y='SALE PRICE', hue='BOROUGH', data=df, palette='rainbow', ci=None)

plt.title('Sales per Borough from 2016-2017')

plt.legend(loc='right')
plt.figure(figsize=(20,5))

sns.countplot('MONTH SOLD', hue='YEAR SOLD', data=df, palette='Purples_r')
df.columns = [c.replace(' ', '_') for c in df.columns]
from collections import Counter

NEIGHBORHOOD = list(dict(Counter(df.NEIGHBORHOOD).most_common(20)).keys())



avg_sale_prices = []

for i in NEIGHBORHOOD:

    avg_price = np.mean(df.SALE_PRICE[df.NEIGHBORHOOD == i])

    avg_sale_prices.append(avg_price)
plt.figure(figsize=(20,8))

sns.barplot(x= avg_sale_prices, y= NEIGHBORHOOD , ci=None)

plt.title('Average House Price in the top 20 neighborhoods')
plt.subplots(figsize=(25,15))

wordcloud = WordCloud(

                          background_color='Black',

                          width=1920,

                          height=1080

                         ).generate(" ".join(df.NEIGHBORHOOD))

plt.imshow(wordcloud)

plt.axis('off')

plt.savefig('cast.png')

plt.show()