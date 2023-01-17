import numpy as np

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns

import statsmodels.api as sm



sns.set()

rand_state=1000
df = pd.read_csv('../input/nyc-property-sales/nyc-rolling-sales.csv')

df_raw = df

df.head()
df.info()
# First Let's remove  irrelavant columns: 

df.drop(["Unnamed: 0"], axis=1, inplace=True)
# constructing the date time variable



df['SALE DATE']= pd.to_datetime(df['SALE DATE'], errors='coerce')
df['sale_year'] = pd.DatetimeIndex(df['SALE DATE']).year.astype("category")

df['sale_month'] = pd.DatetimeIndex(df['SALE DATE']).month.astype("category")

pd.crosstab(df['sale_month'],df['sale_year'])
# constructing the numerical variables:

numeric = ["RESIDENTIAL UNITS","COMMERCIAL UNITS","TOTAL UNITS", "LAND SQUARE FEET" , "GROSS SQUARE FEET","SALE PRICE" ]



for col in numeric: 

    df[col] = pd.to_numeric(df[col], errors='coerce') # coercing errors to NAs
# constructing the categorical variables:

categorical = ["BOROUGH","NEIGHBORHOOD",'BUILDING CLASS CATEGORY', 'TAX CLASS AT PRESENT', 'BUILDING CLASS AT PRESENT','ZIP CODE', 'YEAR BUILT', 'BUILDING CLASS AT TIME OF SALE', 'TAX CLASS AT TIME OF SALE']



for col in categorical: 

    df[col] = df[col].astype("category")
df.info()
df.isna().sum()
df.replace(' ',np.nan, inplace=True)

df.isna().sum() /len(df) *100
plt.figure(figsize=(10,7))

sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='viridis')
df.drop(["EASE-MENT","APARTMENT NUMBER"], axis=1, inplace=True)
# What should we do with the LAND and GROSS sqrf? there are more than 30% of missing data here! One way is to get rid of all the NAs. 

# but this is not the best solution! 

df=df.dropna() 
# finally check if there is any duplicated value:

sum(df.duplicated())
df.drop_duplicates(inplace=True)
temp = df.copy()

for cols in temp.columns:

    temp[cols]=pd.to_numeric(temp[cols], errors='coerce') 

    

temp.info()

temp.describe().T
df[(df['SALE PRICE']<10000) | (df['SALE PRICE']>10000000)]['SALE PRICE'].count() /len(df)
df2= df[(df['SALE PRICE']>10000) & (df['SALE PRICE']<10000000)].copy()

df2['SALE PRICE'].describe()
plt.figure(figsize=(12,6))

sns.distplot(df2['SALE PRICE'], kde=True, bins=50, rug=True)

plt.show()
df2= df2[(df2['SALE PRICE']<4000000)]

plt.figure(figsize=(12,6))

sns.distplot(df2['SALE PRICE'], kde=True, bins=50, rug=True)

plt.show()

df2[df2['YEAR BUILT']==0]['YEAR BUILT'].count()
df3=df2[df2['YEAR BUILT']!=0].copy()

sns.distplot(df3['YEAR BUILT'], bins=50, rug=True)

plt.show()
df3[df3['TOTAL UNITS']==0]['TOTAL UNITS'].count()
df4=df3[df3['TOTAL UNITS']!=0].copy()

sns.distplot(df4['TOTAL UNITS'], bins=50, rug=True)

plt.show()
df4.describe().T
df4.info()
df4.drop(['BLOCK','LOT','ADDRESS'], axis=1, inplace=True)
#'1':'Manhattan', '2':'Bronx', '3': 'Brooklyn', '4':'Queens','5':'Staten Island'

df4['BOROUGH']= df4['BOROUGH'].map({1:'Manhattan', 2:'Bronx', 3: 'Brooklyn', 4:'Queens',5:'Staten Island'})

df4.head()
# some other visualizations: 

df_bar =df4[['BOROUGH', 'SALE PRICE']].groupby(by='BOROUGH').mean().sort_values(by='SALE PRICE', ascending=True).reset_index()

df_bar
plt.figure(figsize=(12,6))

sns.barplot(y = 'BOROUGH', x = 'SALE PRICE', data = df_bar )

plt.title('Average SALE PRICE on each BOROUGH')

plt.show()
plt.figure(figsize=(12,6))

sns.boxplot(y = 'BOROUGH', x = 'SALE PRICE', data = df4 )

plt.title('Box plots for SALE PRICE on each BOROUGH')

plt.show()
df_bar=df4[['sale_month', 'SALE PRICE']].groupby(by='sale_month').count().sort_values(by='sale_month', ascending=True).reset_index()

df_bar.columns.values[1]='Sales_count'

df_bar
plt.figure(figsize=(12,6))

sns.barplot(y = 'sale_month', x = 'Sales_count', data = df_bar )

plt.title('count SALEs by each month')

plt.show()