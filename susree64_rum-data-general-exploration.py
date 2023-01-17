import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
import os
print(os.listdir("../input"))

# Read the data into dataframe
df = pd.read_csv("../input/rum_data.csv")
df.head()
# Required to clean up and change the data format of price, ratings to proper formats
def mod(str):
    str = str.replace(',','.')
    return(float(str))
convert = lambda x:mod(x)
# First Remove NaNs 
df = df.dropna().reset_index(drop = True)
# Apply modifications for format change of columns data
df['Price'] = df['Price'].apply(mod)
df['Rating'] = df['Rating'].apply(convert)
df.head()
print("Number of observations or rows of data : ", df.shape[0])
print("Number features or colmns of data : ", df.shape[1])
sns.pairplot(df)
plt.show()
country_rating = df[['Country', 'Rating']].groupby('Country').mean().sort_values('Rating',ascending = False).reset_index()
plt.figure(figsize = (20,20))
sns.barplot(y = 'Country',x =  'Rating', data = country_rating)
plt.title('Countries & Rating')
plt.xticks(rotation = 90)
plt.show()
print('Top 5 Countires which produce highest averaging rating Rum are')
print(country_rating.head(5))
lable_rating = df[['Label','Rating']].sort_values('Rating', ascending = False).head(10)
plt.figure(figsize = (10,10))
sns.barplot(x = 'Rating', y = 'Label', data = lable_rating)
plt.title('TOP 10 Labels that are Hihgly rated ')
plt.show()

print('Top 10 Lables')
print(lable_rating.head(10))
category_rating = df[['Category', 'Rating']].groupby('Category').mean().sort_values('Rating',ascending = False). reset_index()
print(category_rating)
rating_price = df[['Label','Rating', 'Price']]

# Regression Plot of Ratings and Price 
plt.figure (figsize = (15, 8))
sns.regplot(x = 'Rating', y = 'Price', data = rating_price, marker = '+', color = 'g')
plt.title('Ratings and Price Regression plot')
plt.show()
print( 'Top 5 Highly priced Labels')
print(rating_price.sort_values('Price',ascending = False).head(5))
rating_price.sort_values('Rating',ascending = False)

# The Below Picks up the groped min values 
least_prices = rating_price.groupby('Rating')['Price'].transform('min')
# Get all the rows, where the least_prices are in 
lowprice_highrated = rating_price.loc[rating_price['Price'].isin(least_prices)]
lowprice_highrated = lowprice_highrated.sort_values('Rating', ascending = False).reset_index(drop = True)
lowprice_highrated

highest_prices = rating_price.groupby('Rating')['Price'].transform('max')
# Get all the rows, where the least_prices are in 
highprice_highrated = rating_price.loc[rating_price['Price'].isin(highest_prices)]
highprice_highrated = highprice_highrated.sort_values('Rating', ascending = False).reset_index(drop = True)
highprice_highrated
plt.figure(figsize  = (20,10))
plt.rcParams.update({'font.size': 18})
plt.plot( highprice_highrated['Rating'],highprice_highrated['Price'], '-o', markersize= 8)
plt.plot(lowprice_highrated['Rating'],lowprice_highrated['Price'],'-*' , markersize=8)
plt.title('Comparision of Prices of Low Priced and High Priced Labels with the same Ratings')
#Changing the Legend text 
# Gain acess to legend object and change the text property
L=plt.legend()
L.get_texts()[0].set_text('High Price Lables')
L.get_texts()[1].set_text('Low Price Lables')
plt.show()
df1 = df[df['Sugar'] !=0]
plt.figure(figsize = (20,10))
df1 = df1.sort_values('Rating')
plt.plot(df1['Rating'], df1['Price'], '-o', markersize= 8)
plt.plot(df1['Rating'], df1['Sugar'],'-*', markersize= 8)
plt.title('Rating Vs Price Rating Vs Price')
L=plt.legend()
L.get_texts()[0].set_text('Price')
L.get_texts()[1].set_text('Sugar')
plt.show()
