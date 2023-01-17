#importing the required libraries

import pandas as pd

import numpy as np



#viz Libraries

import matplotlib.pyplot as plt



plt.style.use('ggplot')

import seaborn as sns



#warnings

import warnings

warnings.filterwarnings("ignore")



#datetime

import datetime as dt



#StandardSccaler

from sklearn.preprocessing import StandardScaler



#KMeans

from sklearn.cluster import KMeans



#file directoryy

import os


for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



#reading the data

df = pd.read_csv('../input/sample-sales-data/sales_data_sample.csv', encoding = 'unicode_escape')
df.shape #Dimensions of the data
df.head() #Glimpse of the data
#Removing the variables which dont add significant value fot the analysis.

to_drop = ['PHONE','ADDRESSLINE1','ADDRESSLINE2','STATE','POSTALCODE']

df = df.drop(to_drop, axis=1)
df.isnull().sum()
df.dtypes
df['ORDERDATE'] = pd.to_datetime(df['ORDERDATE'])
quant_vars = ['QUANTITYORDERED','PRICEEACH','SALES','MSRP']

df[quant_vars].describe()
plt.figure(figsize=(9,6))

sns.distplot(df['QUANTITYORDERED'])

plt.title('Order Quantity Distribution')

plt.xlabel('Quantity Ordered')

plt.ylabel('Frequency')

plt.show()
plt.figure(figsize=(9,6))

sns.distplot(df['PRICEEACH'])

plt.title('Price Distribution')

plt.xlabel('Price Ordered')

plt.ylabel('Frequency')

plt.show()
plt.figure(figsize=(9,6))

sns.distplot(df['SALES'])

plt.title('Sales Distribution')

plt.xlabel('Sales')

plt.ylabel('Frequency')

plt.show()
df['STATUS'].value_counts(normalize = True)
df.groupby(['YEAR_ID'])['MONTH_ID'].nunique()
plt.figure(figsize=(9,6))

df['DEALSIZE'].value_counts(normalize = True).plot(kind = 'bar')

plt.title('DealSize distribution')

plt.xlabel('Deal Size')

plt.ylabel('% Proportion')

plt.show()
#Annual Revenue

plt.figure(figsize=(9,6))

df.groupby(['YEAR_ID'])['SALES'].sum().plot()

plt.xlabel('Year')

plt.ylabel('Revenue')

plt.title('Annual Revenue')

plt.xticks(np.arange(2003,2006,1))

plt.show()
#Monthly Revenue

plt.figure(figsize=(9,6))



monthly_revenue = df.groupby(['YEAR_ID','MONTH_ID'])['SALES'].sum().reset_index()

monthly_revenue

sns.lineplot(x="MONTH_ID", y="SALES",hue="YEAR_ID", data=monthly_revenue)

plt.xlabel('Month')

plt.ylabel('Sales')

plt.title('Monthly Revenue')

plt.show()
monthly_revenue['MONTHLY GROWTH'] = monthly_revenue['SALES'].pct_change()
monthly_revenue.head()
#Monthly Sales Growth Rate

plt.figure(figsize=(9,6))

sns.lineplot(x="MONTH_ID", y="MONTHLY GROWTH",hue="YEAR_ID", data=monthly_revenue)

plt.xlabel('Month')

plt.ylabel('Sales')

plt.title('Monthly Sales Growth Rate')

plt.show()
plt.figure(figsize=(9,6))

top_cities = df.groupby(['COUNTRY'])['SALES'].sum().sort_values(ascending=False)

top_cities.plot(kind = 'bar')

plt.title('Top 10 countries by Sales')

plt.xlabel('Country')

plt.ylabel('Total Sales')

plt.show()
#plt.figure(figsize=(10,8))

df['YEAR_MONTH'] = df['YEAR_ID'].map(str)+df['MONTH_ID'].map(str).map(lambda x: x.rjust(2,'0'))

monthly_active = df.groupby(['YEAR_MONTH'])['CUSTOMERNAME'].nunique().reset_index()

monthly_active.plot(kind='bar',x='YEAR_MONTH',y='CUSTOMERNAME')

#plt.figure(figsize=(10,8))

plt.title('Monthly Active Customers')

plt.xlabel('Month/Year')

plt.ylabel('Number of Unique Customers')

plt.xticks(rotation=90)

#plt.figure(figsize=(10,8))

plt.show()
#Average Sales per Order

average_revenue = df.groupby(['YEAR_ID','MONTH_ID'])['SALES'].mean().reset_index()

plt.figure(figsize=(10,6))

sns.lineplot(x="MONTH_ID", y="SALES",hue="YEAR_ID", data=average_revenue)

plt.xlabel('Month')

plt.ylabel('Average Sales')

plt.title('Average Sales per Order')

plt.show()
#New Customers Growth Rate

df_first_purchase = df.groupby('CUSTOMERNAME').YEAR_MONTH.min().reset_index()

df_first_purchase.columns = ['CUSTOMERNAME','FirstPurchaseDate']



plt.figure(figsize=(10,6))

df_first_purchase.groupby(['FirstPurchaseDate'])['CUSTOMERNAME'].nunique().pct_change().plot(kind='bar')

plt.title('New Customers Growth Rate')

plt.xlabel('YearMonth')

plt.ylabel('Percentage Growth Rate')

plt.show()
df['ORDERDATE'] = [d.date() for d in df['ORDERDATE']]

df.head()
# Calculate Recency, Frequency and Monetary value for each customer

snapshot_date = df['ORDERDATE'].max() + dt.timedelta(days=1) #latest date in the data set

df_RFM = df.groupby(['CUSTOMERNAME']).agg({

    'ORDERDATE': lambda x: (snapshot_date - x.max()).days,

    'ORDERNUMBER': 'count',

    'SALES':'sum'})



#Renaming the columns

df_RFM.rename(columns={'ORDERDATE': 'Recency',

                   'ORDERNUMBER': 'Frequency',

                   'SALES': 'MonetaryValue'}, inplace=True)

df_RFM.head()
#Dividing into segments



# Create a spend quartile with 4 groups - a range between 1 and 5

MonetaryValue_quartile = pd.qcut(df_RFM['MonetaryValue'], q=4, labels=range(1,5))

Recency_quartile = pd.qcut(df_RFM['Recency'], q=4, labels=list(range(4, 0, -1)))

Frequency_quartile = pd.qcut(df_RFM['Frequency'], q=4, labels=range(1,5))





# Assign the quartile values to the Spend_Quartile column in data

df_RFM['R'] = Recency_quartile

df_RFM['F'] = Frequency_quartile

df_RFM['M'] = MonetaryValue_quartile



#df_RFM[['MonetaryValue_Quartile','Recency_quartile','Frequency_quartile']] = [MonetaryValue_quartile,Recency_quartile,Frequency_quartile]



# Print data with sorted Spend values

#print(df_RFM.sort_values('MonetaryValue'))



df_RFM.head()
# Calculate RFM_Score

df_RFM['RFM_Score'] = df_RFM[['R','F','M']].sum(axis=1)

df_RFM.head()
#Naming Levels

# Define rfm_level function

def rfm_level(df):

    if np.bool(df['RFM_Score'] >= 10):

        return 'High Value Customer'

    elif np.bool((df['RFM_Score'] < 10) & (df['RFM_Score'] >= 6)):

        return 'Mid Value Customer'

    else:

        return 'Low Value Customer'



# Create a new variable RFM_Level

df_RFM['RFM_Level'] = df_RFM.apply(rfm_level, axis=1)



# Print the header with top 5 rows to the console

df_RFM.head()
plt.figure(figsize=(10,6))

df_RFM['RFM_Level'].value_counts(normalize = True).plot(kind='bar')

plt.title('RFM_level Distribution')

plt.xlabel('RFM_Level')

plt.ylabel('% Proportion')

plt.show()
#Analyzing customer segments

# Calculate average values for each RFM_Level, and return a size of each segment 

rfm_level_agg = df_RFM.groupby(['RFM_Level']).agg({

    'Recency': 'mean',

    'Frequency': 'mean',

    'MonetaryValue':['mean','count']}).round(1)



# Print the aggregated dataset

print(rfm_level_agg)
data = df_RFM[['Recency','Frequency','MonetaryValue']]

data.head()
plt.figure(figsize=(10,6))



plt.subplot(1,3,1)

data['Recency'].plot(kind='hist')

plt.title('Recency')



plt.subplot(1,3,2)

data['Frequency'].plot(kind='hist')

plt.title('Frequency')



plt.subplot(1,3,3)

data['MonetaryValue'].plot(kind='hist')

plt.xticks(rotation = 90)

plt.title('MonetaryValue')



plt.tight_layout()

plt.show()
data_log = np.log(data)
data_log.head()
plt.figure(figsize=(10,6))



#plt.subplot(1,3,1)

sns.distplot(data_log['Recency'],label='Recency')



#plt.subplot(1,3,1)

sns.distplot(data_log['Frequency'],label='Frequency')



#plt.subplot(1,3,1)

sns.distplot(data_log['MonetaryValue'],label='MonetaryValue')



plt.title('Distribution of Recency, Frequency and MonetaryValue after Log Transformation')

plt.legend()

plt.show()
# Initialize a scaler

scaler = StandardScaler()



# Fit the scaler

scaler.fit(data_log)



# Scale and center the data

data_normalized = scaler.transform(data_log)



# Create a pandas DataFrame

data_normalized = pd.DataFrame(data_normalized, index=data_log.index, columns=data_log.columns)



# Print summary statistics

data_normalized.describe().round(2)
# Fit KMeans and calculate SSE for each k

sse={}

for k in range(1, 21):

    kmeans = KMeans(n_clusters=k, random_state=1)

    kmeans.fit(data_normalized)

    sse[k] = kmeans.inertia_ 



    

plt.figure(figsize=(10,6))

# Add the plot title "The Elbow Method"

plt.title('The Elbow Method')



# Add X-axis label "k"

plt.xlabel('k')



# Add Y-axis label "SSE"

plt.ylabel('SSE')



# Plot SSE values for each key in the dictionary

sns.pointplot(x=list(sse.keys()), y=list(sse.values()))

plt.text(4.5,60,"Largest Angle",bbox=dict(facecolor='lightgreen', alpha=0.5))

plt.show()
# Initialize KMeans

kmeans = KMeans(n_clusters=5, random_state=1) 



# Fit k-means clustering on the normalized data set

kmeans.fit(data_normalized)



# Extract cluster labels

cluster_labels = kmeans.labels_



# Assigning Cluster Labels to Raw Data

# Create a DataFrame by adding a new cluster label column

data_rfm = data.assign(Cluster=cluster_labels)

data_rfm.head()
# Group the data by cluster

grouped = data_rfm.groupby(['Cluster'])



# Calculate average RFM values and segment sizes per cluster value

grouped.agg({

    'Recency': 'mean',

    'Frequency': 'mean',

    'MonetaryValue': ['mean', 'count']

  }).round(1)

data_rfm_melt = pd.melt(data_rfm.reset_index(), id_vars=['CUSTOMERNAME', 'Cluster'],

                        value_vars=['Recency', 'Frequency', 'MonetaryValue'], 

                        var_name='Metric', value_name='Value')



plt.figure(figsize=(10,6))

# Add the plot title

plt.title('Snake plot of normalized variables')



# Add the x axis label

plt.xlabel('Metric')



# Add the y axis label

plt.ylabel('Value')



# Plot a line for each value of the cluster variable

sns.lineplot(data=data_rfm_melt, x='Metric', y='Value', hue='Cluster')

plt.show()

# Calculate average RFM values for each cluster

cluster_avg = data_rfm.groupby(['Cluster']).mean() 

print(cluster_avg)
# Calculate average RFM values for the total customer population

population_avg = data.mean()

print(population_avg)
# Calculate relative importance of cluster's attribute value compared to population

relative_imp = cluster_avg / population_avg - 1



# Print relative importance score rounded to 2 decimals

print(relative_imp.round(2))
#Plot Relative Importance



# Initialize a plot with a figure size of 8 by 2 inches 

plt.figure(figsize=(8, 2))



# Add the plot title

plt.title('Relative importance of attributes')



# Plot the heatmap

sns.heatmap(data=relative_imp, annot=True, fmt='.2f', cmap='RdYlGn')

plt.show()