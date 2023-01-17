#import required libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import datetime as dt



from sklearn.preprocessing import StandardScaler

from sklearn.cluster import KMeans

from sklearn.metrics import silhouette_samples, silhouette_score



from scipy.cluster import hierarchy

from scipy.spatial import distance_matrix

from sklearn.cluster import AgglomerativeClustering
# read excel file into dataframe

df = pd.read_csv('../input/data.csv',encoding="ISO-8859-1")
# 5 records of datframe

df.head(5)
# shape of df before cleaning 

df.shape
# checking null values in df

missing_df = df.isnull()
# column wise null values

for col in missing_df.columns:

    print(missing_df[col].value_counts(),'\n')
# drop rows where customer id is null

df.dropna(subset=['CustomerID'], axis=0, inplace=True)

df.reset_index(drop=True, inplace=True)
df.shape
## check for duplicates transactions (entries)

df[df.duplicated(keep='first')].head(10)           #5225

df.duplicated().sum()
df.drop_duplicates(inplace=True)
df.shape                            #406829-5225
# statistical summary fro all types of variables i.e numeric and categorical

df.describe(include='all')
# check for invoice number

# InvoiceNo: Invoice number. Nominal, a 6-digit integral number uniquely assigned to each transaction. 

# If this code starts with letter 'c', it indicates a cancellation.

df['cancelled_order'] = df['InvoiceNo'].apply(lambda x : int('C' in str(x)))
df['cancelled_order'].value_counts()
nocancel_df = df[df['cancelled_order']==0]

cancel_df = df[df['cancelled_order']==1]
print(nocancel_df.shape)

print(cancel_df.shape)

print(df.shape)
# StockCode: Product (item) code. Nominal, a 5-digit integral number uniquely assigned to each distinct product.

# check for data where stockcode is not int

df['InValidStock']= df['StockCode'].str.contains('^[a-zA-Z]+', regex=True)
# storing invalid stockcodes in the list

invalid_stocks = df[df['InValidStock']==True]['StockCode'].unique().tolist()

invalid_stocks 
# checking rows which have invalid stockcodes

df[df['StockCode'].isin(invalid_stocks)].count()
# dataframe with valid stockcodes

df = df[~df['StockCode'].isin(invalid_stocks)]
df.shape
df.drop(columns=['InValidStock'], axis=1, inplace=True)
# shape of dataframe after cleaning

df.shape
# Quantity: The quantities of each product (item) per transaction. Numeric.

df[df['Quantity']<0].shape
# only cancelled orders have quantity in negative

df[df['Quantity']<0]['cancelled_order'].value_counts()
#UnitPrice: Unit price. Numeric, Product price per unit in sterling.

#free stuff

df[df['UnitPrice']==0].count()
# deriving total price for each line item level bill

df['Total_Price'] = df['UnitPrice']*df['Quantity']
df[df['cancelled_order']==1].head(5)
# Top selling products

product_df = df.groupby(['StockCode','Description']).agg({'Total_Price' : np.sum, 'InvoiceNo' : pd.Series.nunique, 

                                                          'Quantity' : np.sum, 'CustomerID' : pd.Series.nunique})

product_df.sort_values(by=['Total_Price', 'InvoiceNo','Quantity', 'CustomerID'], ascending=False, inplace=True)
product_df.head(10)
top_10_product = product_df.head(10)

plt.figure(figsize=(8,8))

top_10_product.plot(kind='bar')
# Top countries as per number of transaction

# univariate analysis for categorical variables

plt.figure(figsize=(10,10))

sns.countplot(y='Country', data=df)
# function to return first day of every date 

def get_month(x):

    return dt.datetime(x.year, x.month, 1)
df['InvoiceDate']= pd.to_datetime(df['InvoiceDate']) 

df['Invoice_Month'] = df['InvoiceDate'].apply(get_month)
df.sort_values(by=['InvoiceDate'], ascending=False).head(5)
# grouping data by customer id

cust_grouping = df.groupby('CustomerID')['Invoice_Month']
# adding cohort month for each customer by taking minimum bill date

df['Cohort_Month'] = cust_grouping.transform('min')
# funciton to find year, moth, day of each date

def get_date_int(df, col):

    year = df[col].dt.year

    month = df[col].dt.month

    day = df[col].dt.day

    return year, month, day
invoice_year, invoice_month, _ =  get_date_int(df, 'Invoice_Month')

cohort_year, cohort_month, _ =  get_date_int(df, 'Cohort_Month')
year_diff = invoice_year-cohort_year

month_diff = invoice_month-cohort_month
# calculating cohort index 

df['cohort_index'] = year_diff*12 + month_diff + 1
# checking data

df.head(5)
# grouping data by cohort month and cohort index

cust_grouping = df.groupby(['Cohort_Month','cohort_index'])



# aggregating data for sales, bills, customers, quantities

cohort_data = cust_grouping.agg({'Total_Price':np.sum, 'Quantity':np.sum, 'CustomerID':pd.Series.nunique, 

                                 'InvoiceNo':pd.Series.nunique})
cohort_data = cohort_data.reset_index()

cohort_data['Cohort_Month'] = cohort_data['Cohort_Month'].dt.date
# checking data 

cohort_data[0:10]
# rename columns for better understanding 

cohort_data.rename(columns={'Total_Price':'Total_Sales', 'Quantity':'Total_Quantity',

                           'CustomerID':'Total_Customers', 'InvoiceNo':'Total_Invoices'}, inplace=True)
# pivot data for cohort analysis

cohort_count = cohort_data.pivot(index='Cohort_Month', columns='cohort_index')                 #cohort_count2
# explore cohort data

cohort_count
cohort_count_cust = cohort_count['Total_Customers']
# taking cohort index 1 data fro retention 

cohort_sizes = cohort_count_cust.iloc[:,0]
# finding retention 

retention = cohort_count_cust.divide(cohort_sizes, axis=0)
retention.round(3)*100
# visualize the retention cohort data

plt.figure(figsize=(10,8))

plt.title('Retension Rates')

sns.heatmap(data=retention, annot=True, fmt='.0%', cmap='viridis', vmin=0.0, vmax=0.5)
retention
# data of first 3 cohort months

retention.transpose().iloc[:,:3]
# Visualization of cohort index for first three cohort month

plt.rcParams['figure.figsize'] = 15, 10

plt.rcParams['font.size'] = 12

retention.transpose().iloc[:,:3].plot()

plt.title('Customer Retention')

plt.xticks(range(1, 14))

plt.xlim(1, 14)

plt.ylabel('% of Cohort Purchasing')

plt.show()
# transactional level analysis for all months

invoice_agg = df.groupby('Invoice_Month')
# aggregation of data for monthwise analysis of retail KPIs

agg_data = invoice_agg.agg({'Total_Price':np.sum, 'Quantity':np.sum, 'CustomerID':pd.Series.nunique, 

                                 'InvoiceNo':pd.Series.nunique})
agg_data = agg_data.reset_index()

agg_data
# rename coulmns for better understanding

agg_data.rename(columns={'Total_Price':'Total_Sales', 'Quantity':'Total_Quantity',

                           'CustomerID':'Total_Customers', 'InvoiceNo':'Total_Invoices'}, inplace=True)
agg_data
agg_data['Invoice_Month'] = agg_data['Invoice_Month'].dt.date
# monthwise trend of sales and quantities

ax = plt.gca()

agg_data.plot.line(x='Invoice_Month', y='Total_Sales', ax=ax)

agg_data.plot.line(x='Invoice_Month', y='Total_Quantity', ax=ax)

plt.show()
# monthwise trend of customers and bills

ax = plt.gca()

agg_data.plot.line(x='Invoice_Month', y='Total_Customers', ax=ax)

agg_data.plot.line(x='Invoice_Month', y='Total_Invoices', ax=ax)

plt.show()
df.head(5)
print(min(df['InvoiceDate']), max(df['InvoiceDate']))
snapshot_date = max(df['InvoiceDate']) + dt.timedelta(days=1)

snapshot_date
# aggregation of data at customer level for RFM segmentation

customer_view = df.groupby(['CustomerID']).agg({'InvoiceDate':lambda x: (snapshot_date-x.max()).days, 

                                                'InvoiceNo': pd.Series.nunique, 

                                                'Total_Price': np.sum})
customer_view.rename(columns = {'InvoiceDate':'Recency',

                               'InvoiceNo':'Frequency',

                               'Total_Price':'Monetary'}, inplace=True)

customer_view.head(5)
# checking the data

customer_view.index

customer_view.loc[12820]
# Quantiles for RFM segmentation

quantiles = customer_view.quantile(q=[0.25,0.5,0.75])

quantiles
# function for recency score

def RScore(x,p,quan):

    if x<=quan[p][0.25]:

        return 4

    if x<=quan[p][0.5]:

        return 3

    if x<=quan[p][0.75]:

        return 2

    else:

        return 1   
# function for Frequency & monetory score

def FMScore(x,p,quan):

    if x<=quan[p][0.25]:

        return 1

    if x<=quan[p][0.5]:

        return 2

    if x<=quan[p][0.75]:

        return 3

    else:

        return 4   
# mapping quantiles

customer_view['R'] = customer_view['Recency'].apply(RScore,args=('Recency',quantiles,))

customer_view['F'] = customer_view['Frequency'].apply(FMScore,args=('Frequency',quantiles,))

customer_view['M'] = customer_view['Monetary'].apply(FMScore,args=('Monetary',quantiles,))
# counts for each RFM quantile

for col in customer_view.columns:

    if col=='R' or col=='F' or col=='M':

        print(customer_view[col].value_counts())
# checking the data

customer_view.head(5)
# function for RFM Segment

def Join_RFM(x):

    return str(x['R']) + str(x['F']) + str(x['M'])
# map rfm segment

customer_view['RFM_Segment'] = customer_view.apply(Join_RFM, axis=1)
# map rfm score by adding values of R,F,M

customer_view['RFM_Score'] = customer_view[['R','F','M']].sum(axis=1)
customer_view.head(5)
# top 10 group as per RFM_Segment

customer_view.groupby(['RFM_Segment']).size().sort_values(ascending=False)[:10]
# summary of Recency, freq, montory as per RFM score

customer_view.groupby('RFM_Score').agg({'Recency':np.mean, 'Frequency':np.mean, 'Monetary':[np.mean,np.size]}).round(1)
# grouping into named segments

def segment_customer(df):

    if df['RFM_Score']>=9:

        return '1_Gold'

    elif df['RFM_Score']>=5 and df['RFM_Score']<9:

        return '2_Silver'

    else:

        return '3_Bronze'  
customer_view['Segment'] = customer_view.apply(segment_customer, axis=1)
# summary of Recency, freq, montory as per segment

customer_view.groupby('Segment').agg({'Recency':np.mean, 'Frequency':np.mean, 'Monetary':[np.mean,np.size]}).round(1)
# checking for skewness of variables

customer_view.skew(axis=0)
# checking for distribution of variables

sns.distplot(customer_view['Recency'])
# checking for distribution of variables

sns.distplot(customer_view['Frequency'])
# checking for distribution of variables

sns.distplot(customer_view['Monetary'])
#normalize the data

ss = StandardScaler()

ss.fit(customer_view[['R', 'F', 'M']])

customer_view_nor = ss.transform(customer_view[['R', 'F', 'M']])
# Elbow criterion to get the Best KMeans 

ks = range(1,8)

inertias=[]

for k in ks :

    # Create a KMeans clusters

    kc = KMeans(n_clusters=k,random_state=1)

    kc.fit(customer_view_nor)

    inertias.append(kc.inertia_)           #sum of squared distance to closest cluster center



# Plot ks vs inertias

f, ax = plt.subplots(figsize=(15, 8))

plt.plot(ks, inertias, '-o')

plt.xlabel('Number of clusters, k')

plt.ylabel('SSE')

plt.xticks(ks)

plt.title('Elbow criterion Method')

plt.show()
#choosing best model

km = KMeans(n_clusters=3)

km.fit(customer_view_nor)

customer_view['Cluster'] = km.labels_
# aggregated data at cluster level

customer_view.groupby(['Cluster']).agg({'Recency':np.mean,'Frequency':np.mean,'Monetary':[np.mean,np.size]}).round(0)
# data preparation for snake plot

customer_view_nor = pd.DataFrame(customer_view_nor,index=customer_view.index,

                                 columns=['R','F','M'])

customer_view_nor['Cluster'] = customer_view['Cluster']

customer_view_nor['Segment'] = customer_view['Segment']

customer_view_nor.reset_index(inplace = True)
# data preparation for snake plot

cust_melt = pd.melt(customer_view_nor, id_vars=['CustomerID','Cluster','Segment'], 

                    value_vars=['R', 'F', 'M'],

                   var_name='Attribute',

                   value_name='Value')
## snake plot to understand and compare segments

sns.lineplot(x='Attribute', y='Value', hue='Cluster', data=cust_melt)
# Relative importance of segment attributes

cluster_avg = customer_view.groupby('Cluster')['R','F','M'].mean()
population_avg = customer_view[['R','F','M']].mean()
relative_imp = cluster_avg/population_avg - 1
relative_imp.round(2)
# ploting relative importances

plt.figure(figsize=(8,4))

sns.heatmap(data=relative_imp,annot=True, fmt='.2f', cmap='viridis')
segment_avg = customer_view.groupby('Segment')['R','F','M'].mean()

relative_imp = segment_avg/population_avg - 1
plt.figure(figsize=(8,4))

sns.heatmap(data=relative_imp,annot=True, fmt='.2f', cmap='viridis')
# silhouette_score for cluster evaluation

silhouette_score(customer_view_nor[['R','F','M']], customer_view['Cluster'])
ss = StandardScaler()

ss.fit(customer_view[['R', 'F', 'M']])

customer_view_nor_agg = ss.transform(customer_view[['R', 'F', 'M']])
# agg clustering

agg = AgglomerativeClustering(n_clusters=3, linkage='complete')

agg.fit(customer_view_nor_agg)

customer_view['AggCluster'] = agg.labels_
customer_view.head(5)
customer_view.groupby(['AggCluster']).agg({'Recency':np.mean,'Frequency':np.mean,'Monetary':[np.mean,np.size]}).round(0)
customer_view_nor_agg = pd.DataFrame(customer_view_nor_agg,index=customer_view.index,

                                 columns=['R','F','M'])

customer_view_nor_agg['AggCluster'] = customer_view['AggCluster']

customer_view_nor_agg['Segment'] = customer_view['Segment']

customer_view_nor_agg.reset_index(inplace = True)
customer_view_nor_agg.head(5)
## snake plot to understand and compare segments

cust_melt_agg = pd.melt(customer_view_nor_agg, id_vars=['CustomerID','AggCluster','Segment'], 

                    value_vars=['R', 'F', 'M'],

                   var_name='Attribute',

                   value_name='Value')
sns.lineplot(x='Attribute', y='Value', hue='AggCluster', data=cust_melt_agg)
dis_mat = distance_matrix(customer_view_nor_agg[['R','F','M']], customer_view_nor_agg[['R','F','M']])

z = hierarchy.linkage(dis_mat, 'complete')

dendro = hierarchy.dendrogram(z)
silhouette_score(customer_view_nor_agg[['R','F','M']], customer_view['AggCluster'])