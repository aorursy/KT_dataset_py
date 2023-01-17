#installation of libraries

import pandas as pd

import numpy as np

import seaborn as sns

from sklearn.cluster import KMeans



#to display all columns and rows:

pd.set_option('display.max_columns', None); pd.set_option('display.max_rows', None);



#we determined how many numbers to show after comma

pd.set_option('display.float_format', lambda x: '%.0f' % x)

import matplotlib.pyplot as plt
#calling the dataset

df = pd.read_csv("../input/online-retail-ii-uci/online_retail_II.csv")
#selection of the first 5 observations

df.head() 
#ranking of the most ordered products

df.groupby("Description").agg({"Quantity":"sum"}).sort_values("Quantity", ascending = False).head()
#how many invoices are there in the data set

df["Invoice"].nunique()
#which are the most expensive products

df.sort_values("Price", ascending = False).head()
#top 5 countries with the highest number of orders

df["Country"].value_counts().head()
#total spending was added as a column

df['TotalPrice'] = df['Price']*df['Quantity']
#which countries did we get the most income from

df.groupby("Country").agg({"TotalPrice":"sum"}).sort_values("TotalPrice", ascending = False).head()
#oldest shopping date

df["InvoiceDate"].min() 
#newest shopping date

df["InvoiceDate"].max() 
#to make the assessment easier, today's date is set as January 1, 2012.  

today = pd.datetime(2012,1,1) 

today
#changing the data type of the order date

df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
#taking values greater than 0, this will be easier in terms of evaluation

df = df[df['Quantity'] > 0]

df = df[df['TotalPrice'] > 0]
#removal of observation units with missing data from df

df.dropna(inplace = True) 
#check for missing values in the dataset

df.isnull().sum(axis=0)
#size information

df.shape 
df.describe([0.01,0.05,0.10,0.25,0.50,0.75,0.90,0.95, 0.99]).T

#explanatory statistics values of the observation units corresponding to the specified percentages

#processing according to numerical variables
#customer distribution by country

country_cust_data=df[['Country','Customer ID']].drop_duplicates()

country_cust_data.groupby(['Country'])['Customer ID'].aggregate('count').reset_index().sort_values('Customer ID', ascending=False)
#keep only United Kingdom data

df_uk = df.query("Country=='United Kingdom'").reset_index(drop=True)

df_uk.head()
df.head()
df.info() 

#dataframe's index dtype and column dtypes, non-null values and memory usage information
#finding Recency and Monetary values.

df_x = df.groupby('Customer ID').agg({'TotalPrice': lambda x: x.sum(), #monetary value

                                        'InvoiceDate': lambda x: (today - x.max()).days}) #recency value

#x.max()).days; last shopping date of customers
df_y = df.groupby(['Customer ID','Invoice']).agg({'TotalPrice': lambda x: x.sum()})

df_z = df_y.groupby('Customer ID').agg({'TotalPrice': lambda x: len(x)}) 

#finding the frequency value per capita
#creating the RFM table

rfm_table= pd.merge(df_x,df_z, on='Customer ID')
#determination of column names

rfm_table.rename(columns= {'InvoiceDate': 'Recency',

                          'TotalPrice_y': 'Frequency',

                          'TotalPrice_x': 'Monetary'}, inplace= True)
rfm_table.head()
#descriptive statistics for Recency

rfm_table.Recency.describe()
#Recency distribution plot

import seaborn as sns

x = rfm_table['Recency']



ax = sns.distplot(x)
#descriptive statistics for Frequency

rfm_table.Frequency.describe()
#Frequency distribution plot, taking observations which have frequency less than 1000

import seaborn as sns

x = rfm_table.query('Frequency < 1000')['Frequency']



ax = sns.distplot(x)
#descriptive statistics for Monetary

rfm_table.Monetary.describe()
#Monateray distribution plot, taking observations which have monetary value less than 10000

import seaborn as sns

x = rfm_table.query('Monetary < 10000')['Monetary']



ax = sns.distplot(x)
#Split into four segments using quantiles

quantiles = rfm_table.quantile(q=[0.25,0.5,0.75])

quantiles = quantiles.to_dict()
quantiles
#conversion process

from sklearn.preprocessing import MinMaxScaler

min_max_scaler = MinMaxScaler((0,1))

x_scaled = min_max_scaler.fit_transform(rfm_table)

data_scaled = pd.DataFrame(x_scaled)
df[0:10]
plt.figure(figsize=(8,6))

wcss = []

for i in range(1, 11):

    kmeans = KMeans(n_clusters = i, init = 'k-means++',n_init=10, max_iter = 300)

    kmeans.fit(data_scaled)

    wcss.append(kmeans.inertia_)

plt.plot(range(1, 11), wcss)

plt.title('The Elbow Method')

plt.xlabel('Number of clusters')

plt.ylabel('WCSS')

plt.show()
kmeans = KMeans(n_clusters = 4, init='k-means++', n_init =10,max_iter = 300)

kmeans.fit(data_scaled)

cluster = kmeans.predict(data_scaled)

#init = 'k-means ++' this makes it work faster
d_frame = pd.DataFrame(rfm_table)

d_frame['cluster_no'] = cluster

d_frame['cluster_no'].value_counts() #the number of people per cluster (Custer ID number)
#RFM Table

rfm_table.head()
#cluster average values

d_frame.groupby('cluster_no').mean() 