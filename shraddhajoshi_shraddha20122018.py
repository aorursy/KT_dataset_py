!pip install wordcloud

from wordcloud import WordCloud

import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt 

%matplotlib inline

import seaborn as sns

from scipy import stats

df = pd.read_csv("/Users/HP/Desktop/data.csv",encoding="ISO-8859-1") 

df.head(10)

df.shape #Number of rows and columns:
list(df.columns) # list of the columns of the dataframe named as df

# Checking the data type

df.dtypes

df = df.drop_duplicates()#eliminating null values

df.shape
df.isnull().sum()
sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='viridis')
df = df.dropna()
sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='viridis')
df.isnull().sum()[:10]
df.info()
# change columns tyoe - String to Int type 

df['CustomerID'] = df['CustomerID'].astype('int64')

df.head(10)
df.describe()
df = df[df.Quantity > 0]

df.describe()
plt.figure(figsize=(20,5))

sns.boxplot(x=df['Quantity'],color='Red')

plt.figure(figsize=(20,5))

sns.boxplot(x=df['UnitPrice'],color='Red')

df['Country'].value_counts()

plt.figure(figsize=(10,8))



Country = df.Country.value_counts()[:20]

del Country['United Kingdom']

g = sns.countplot(x='Country', 

                  data=df[df.Country.isin(Country.index.values)],

                 color = 'skyblue')

g.set_title("Countrywise No. of orders", fontsize=20)

g.set_xlabel("Country ", fontsize=15)

g.set_ylabel("Orders", fontsize=15)

g.set_xticklabels(g.get_xticklabels(),rotation=45)



plt.show()
df.UnitPrice.describe()
df['amount_spent'] = df['Quantity'] * df['UnitPrice']

df.head(10)
df.corr()
df.amount_spent.describe()
money_spent = df.groupby(by=['CustomerID','Country'], as_index=False)['amount_spent'].sum()

print('The TOP 5 customers with highest money spent...')

money_spent.sort_values(by='amount_spent', ascending=False).head()
key_words=','.join(df['Description'].str.cat(sep='|').split('|'))
wc=WordCloud(max_words=30,background_color='white').generate(key_words)

plt.figure(figsize=(10,8))

plt.imshow(wc)

plt.show()
# Finding the relations between the variables.

plt.figure(figsize=(10,5))

c= df.corr()

sns.heatmap(c,cmap="gist_earth",annot=True) #BrBG, RdGy, coolwarm

c
group_country_amount_spent = df.groupby('Country')['amount_spent'].sum().sort_values()

del group_country_amount_spent['United Kingdom']



# plot total money spent by each country (without UK)

plt.subplots(figsize=(15,8))

group_country_amount_spent.plot(kind='barh', fontsize=12,color='skyblue' )

plt.xlabel('Money Spent (Dollar)', fontsize=12)

plt.ylabel('Country', fontsize=12)

plt.title('Money Spent by different Countries', fontsize=12)

plt.show()
df['StockCode'].value_counts()[:20]
plt.figure(figsize=(20,10))



StockCode = df.StockCode.value_counts()[:20]



g = sns.countplot(x='StockCode', 

                  data=df[df.StockCode.isin(StockCode.index.values)],

                 color = 'skyblue')

g.set_title("Most Frequent StockCodes", fontsize=20)

g.set_xlabel("StockCode ", fontsize=15)

g.set_ylabel("Counts", fontsize=15)

g.set_xticklabels(g.get_xticklabels(),rotation=45)



plt.show()
orders = df.groupby(by=['CustomerID','Country'], as_index=False)['InvoiceNo'].count()



print('The TOP 5 customers with most number of orders...')

orders.sort_values(by='InvoiceNo', ascending=False).head()
description_counts = df.Description.value_counts().sort_values(ascending=False).iloc[0:25]

plt.figure(figsize=(20,5))

sns.barplot(description_counts.index, description_counts.values, palette="magma_r")

plt.ylabel("Counts")

plt.title("Which product descriptions are most common?");

plt.xticks(rotation=90);
df.Country.nunique()
print("First Quartile, Q1=", df['Quantity'].quantile(0.25))

print("Second Quartile/Median, Q2=", df['Quantity'].median())

print("third Quartile, Q3=", df['Quantity'].quantile(0.75))

df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate']) 
import datetime as dt

def get_month(x): return dt.datetime(x.year, x.month, 1)

df['InvoiceMonth'] = df['InvoiceDate'].apply(get_month)

grouping = df.groupby('CustomerID')['InvoiceMonth']

df['CohortMonth'] = grouping.transform('min')

df.head(10)
def get_date_int(df, column):

    year = df[column].dt.year

    month = df[column].dt.month

    day = df[column].dt.day

    return year, month, day
invoice_year, invoice_month, _ = get_date_int(df, 'InvoiceMonth')

cohort_year, cohort_month, _ = get_date_int(df, 'CohortMonth')
years_diff = invoice_year - cohort_year

months_diff = invoice_month - cohort_month
df['CohortIndex'] = years_diff * 12 + months_diff + 1
## grouping customer berdasarkan masing masing cohort

grouping = df.groupby(['CohortMonth', 'CohortIndex'])

cohort_data = grouping['CustomerID'].apply(pd.Series.nunique)

cohort_data = cohort_data.reset_index()

cohort_counts = cohort_data.pivot(index='CohortMonth', columns='CohortIndex', values='CustomerID')
cohort_sizes = cohort_counts.iloc[:,0]

retention = cohort_counts.divide(cohort_sizes, axis=0)

retention.round(2) * 100
plt.figure(figsize=(15, 8))

plt.title('Retention rates')

sns.heatmap(data = retention,

annot = True,

fmt = '.0%',

vmin = 0.0,

vmax = 0.5,

cmap = 'Reds')

plt.show()
cohort_counts
df['Weekday'] = df["InvoiceDate"].map(lambda x: x.weekday())

df['Day'] = df["InvoiceDate"].map(lambda x: x.day)

df['Hour'] = df["InvoiceDate"].map(lambda x: x.hour)

df['Month']=df["InvoiceDate"].map(lambda x: x.month)

df.insert(loc=2, column='year_month', value=df['InvoiceDate'].map(lambda x: 100*x.year + x.month))

df.head(5)

ax = df.groupby('InvoiceNo')['Hour'].unique().value_counts().iloc[:-1].sort_index().plot(kind='bar',color='skyblue',figsize=(15,6))

ax.set_xlabel('Hour',fontsize=15)

ax.set_ylabel('Number of Orders',fontsize=15)

ax.set_title('Number of orders for different Hours',fontsize=15)

ax.set_xticklabels(range(6,21), rotation='horizontal', fontsize=15)

plt.show()
ax = df.groupby('InvoiceNo')['year_month'].unique().value_counts().sort_index().plot(kind='bar',color='skyblue',figsize=(15,6))

ax.set_xlabel('Month',fontsize=15)

ax.set_ylabel('Number of Orders',fontsize=15)

ax.set_title('Number of orders for different Months',fontsize=15)

ax.set_xticklabels(('Dec_10','Jan_11','Feb_11','Mar_11','Apr_11','May_11','Jun_11','July_11','Aug_11','Sep_11','Oct_11','Nov_11','Dec_11'),  rotation='horizontal', fontsize=15)

plt.show()