import pandas as pd
import numpy as np
import datetime as td   # Change the alias name
from scipy import stats

import matplotlib.pyplot as plt
import seaborn as sns

%matplotlib inline
sns.set_style('darkgrid')
df = pd.read_excel('../input/OnlineRetail.xlsx')
df.columns.tolist()
df.dtypes
len(df)
df.isnull().sum()
df.describe()
df.dropna(inplace= True)
df.isnull().sum()
# length of negative values
print(len(df[(df.Quantity < 0) | (df.UnitPrice < 0)]), '\n')

# remove them
df = df[~(df.Quantity < 0) | (df.UnitPrice < 0)]
plt.figure(figsize=(20, 6))

bp_bins = int(np.sqrt(len(df.Quantity)))

sns.distplot(df.Quantity, bins= bp_bins)

plt.axvline(df.Quantity.mean())
plt.axvline(df.Quantity.median(), linestyle = '--', c = 'r')
plt.axvline(df.Quantity.quantile(0.25), linestyle = ':', c = 'g')
plt.axvline(df.Quantity.quantile(0.75), linestyle = ':', c = 'g')
plt.margins(0.02)

plt.show()
plt.figure(figsize=(20, 6))

sns.distplot(np.log(df.Quantity), bins= 100)

plt.axvline(df.Quantity.mean())
plt.axvline(df.Quantity.median(), linestyle = '--', c = 'r')
plt.axvline(df.Quantity.quantile(0.25), linestyle = ':', c = 'g')
plt.axvline(df.Quantity.quantile(0.75), linestyle = ':', c = 'g')
plt.margins(0.02)

plt.show()
# ECDF Plotting with Percentiles

plt.figure(figsize=(20,6))

percentiles = np.array([10, 25, 50, 75, 99])
percentiles_vars = np.percentile(df.Quantity, percentiles)

x = np.sort(df.Quantity.values)
y = np.arange(1, len(df.Quantity) + 1) / len(df.Quantity)

plt.plot(x, y, '--')
plt.plot(percentiles_vars, percentiles / 100, marker = 'o', markersize = 10, linestyle = 'none')

plt.margins(0.02)

plt.show()
plt.figure(figsize=(20,6))

sns.boxplot(y = df.Quantity, data= df)
plt.show()
# Remove outliers based on median & 1.5 * iqr
q1  = np.percentile(df.Quantity, 25) 
q2  = np.percentile(df.Quantity, 75)
iqr = stats.iqr(df.Quantity)

clean_df = df[(df.Quantity >= q1 - (iqr * 1.5)) & (df.Quantity <= q2 + (iqr * 1.5))].copy()
clean_df.describe()
# Remove 0 values in both Quantity and UnitPrice
clean_df = clean_df[~((clean_df.Quantity == 0) | (clean_df.UnitPrice == 0))]

# Reset index to avoid errors during the process
clean_df.reset_index(drop= True, inplace= True)
print('Lenght of original Dataset: ', len(df))
print('Lenght of cleaned Dataset: ', len(clean_df))
print('Cleaned Dataset is less by: ', len(df) - len(clean_df))
fig, ax = plt.subplots(ncols= 2, figsize= (25,7))

sns.distplot(clean_df.Quantity, ax = ax[0])
sns.boxplot(y = clean_df.Quantity, data= df, ax = ax[1] )

ax[0].axvline(df.Quantity.mean(), linestyle = '--', c = 'r')
ax[0].axvline(df.Quantity.median(), linestyle = ':', c = 'g')
ax[0].axvline(df.Quantity.quantile(0.25), linestyle = ':', c = 'g')
ax[0].axvline(df.Quantity.quantile(0.75), linestyle = ':', c = 'g')

plt.margins(0.02)

plt.show()
clean_df['MonthlyInvoice'] = pd.to_datetime(clean_df.InvoiceDate.dt.strftime('%Y-%m-1'))
clean_df['CohortGroup_byMonth'] = clean_df.groupby('CustomerID')['MonthlyInvoice'].transform('min')
# Check the types
clean_df.dtypes
# Creating the CohortIndex 

def get_year_month(df, col):
    
    year = df[col].dt.year
    month = df[col].dt.month
    day = df[col].dt.day
    
    return year, month, day

invoice_year, invoice_month, invoice_day = get_year_month(clean_df, 'MonthlyInvoice')
cohort_year, cohort_month, cohort_day = get_year_month(clean_df, 'CohortGroup_byMonth')

year_diff = invoice_year - cohort_year
month_diff = invoice_month - cohort_month

clean_df['CohortIndex'] = year_diff * 12 + month_diff + 1
CohortCounts = clean_df.groupby(['CohortGroup_byMonth', 'CohortIndex'])['CustomerID'].apply(pd.Series.nunique).reset_index()
pivot_CohortCounts = CohortCounts.pivot(index= 'CohortGroup_byMonth', columns= 'CohortIndex', values= 'CustomerID')
pivot_CohortCounts
first_CohortSize = pivot_CohortCounts.iloc[:,0]
retention = pivot_CohortCounts.divide(first_CohortSize, axis= 0).round(3) * 100
retention
# Customer's retention heatmap visualization

plt.figure(figsize= (15,8))

sns.heatmap(retention, yticklabels= retention.index.astype('str'), fmt='g', annot= True,  cmap= 'BuGn') # yticklabels to str to avoid displying the time next to the date

plt.show()
avg_purchase_qty = clean_df.groupby(['CohortGroup_byMonth', 'CohortIndex'])['Quantity'].mean().round(1).reset_index()
pivot_avg_purchase_qty = avg_purchase_qty.pivot(index= 'CohortGroup_byMonth', columns= 'CohortIndex', values= 'Quantity')
pivot_avg_purchase_qty
# Customer's Average Purchase Quantity heatmap visualization

plt.figure(figsize= (15,8))

sns.heatmap(pivot_avg_purchase_qty, yticklabels= pivot_avg_purchase_qty.index.astype('str'), fmt='g', annot= True,  cmap= 'BuGn') # yticklabels to str to avoid displying the time next to the date

plt.show()
# Create Total Price column to perform the Average Purchase Power calculation
clean_df['TotalPrice'] = clean_df.UnitPrice * clean_df.Quantity

avg_purchase_power = clean_df.groupby(['CohortGroup_byMonth', 'CohortIndex'])['TotalPrice'].mean().round(1).reset_index()
pivot_avg_purchase_power = avg_purchase_power.pivot(index= 'CohortGroup_byMonth', columns= 'CohortIndex', values= 'TotalPrice')
pivot_avg_purchase_power
# Customer's Average Purchase Power heatmap visualization

plt.figure(figsize= (15,8))

sns.heatmap(pivot_avg_purchase_power, yticklabels= pivot_avg_purchase_power.index.astype('str'), fmt='g', annot= True,  cmap= 'BuGn') # yticklabels to str to avoid displying the time next to the date

plt.show()
# Check the minimum and maximum dates
print('Minimum Date: ', clean_df.InvoiceDate.dt.date.min())
print('Maximum Date: ', clean_df.InvoiceDate.dt.date.max())
# Add 1 day to the max date to perform more realistic calculations
snapDate = clean_df.InvoiceDate.dt.date.max() + td.timedelta(days = 1)
snapDate
RFM_Segmentation = clean_df.groupby('CustomerID').agg( {   'InvoiceDate': lambda x: (snapDate - x.dt.date.max()).days,
                                                           'InvoiceNo': 'count',
                                                           'TotalPrice': 'sum'  })

RFM_Segmentation.rename(columns = {'InvoiceDate': 'Recency',
                                   'InvoiceNo': 'Frequency',
                                   'TotalPrice': 'MonetaryValue'}, inplace = True)

RFM_Segmentation.head()
r_labels = range(5, 0, -1)
f_labels = range(1,6)
m_labels = range(1,6)

r_quartile = pd.qcut( x= RFM_Segmentation.Recency, q= 5, labels= r_labels)
f_quartile = pd.qcut( x= RFM_Segmentation.Frequency, q= 5, labels= f_labels)
m_quartile = pd.qcut( x= RFM_Segmentation.MonetaryValue, q= 5, labels= m_labels)

RFM_Segmentation = RFM_Segmentation.assign(R= r_quartile.values)
RFM_Segmentation = RFM_Segmentation.assign(F= f_quartile.values)
RFM_Segmentation = RFM_Segmentation.assign(M= m_quartile.values)

RFM_Segmentation.head(10)
RFM_Segmentation['RFM_Segment'] = RFM_Segmentation.R.astype(str) + RFM_Segmentation.F.astype(str) + RFM_Segmentation.M.astype(str)
RFM_Segmentation['RFM_Score'] = RFM_Segmentation.R.astype(int) + RFM_Segmentation.F.astype(int) + RFM_Segmentation.M.astype(int)  # OR --> RFM_Segmentation[['R', 'F', 'M']].sum(axis=1)
RFM_Segmentation.head(10)
count = RFM_Segmentation.RFM_Segment.value_counts()
perct = (RFM_Segmentation.RFM_Segment.value_counts(normalize= True).round(3) * 100)

df_basic_summary = pd.DataFrame({'seg_count': count, 'seg_perct': perct}).reset_index().rename(columns = {'index': 'segmentation'})
df_basic_summary[:5]
plt.figure(figsize= (12,6))

_ = sns.barplot( data = df_basic_summary[:10], x = 'segmentation', y = 'seg_count', order= df_basic_summary.sort_values('seg_count', ascending= False)['segmentation'][:10])

annotations = df_basic_summary.seg_perct.round(1).tolist()
counter = 0

for p in _.patches:
    height = p.get_height()
    _.text(p.get_x() + p.get_width()/2., height + 6, '({}%)'.format(annotations[counter]), ha="center", va='center', fontsize=10)
    counter += 1

_.set_title('Customers RFM Segmentation - Top 10', pad = 10, weight= 'bold')
_.set_xlabel('RFM Segmentation', weight= 'bold')
_.set_ylabel('Counts + Contribution', weight= 'bold')

plt.show()
# Customized Segmentation Function
def func_customized_segmentation(x):
    
    if x > 10:
        return 'Gold'
    elif x > 5:
        return 'Silver'
    else:
        return 'Bronze'

# Assign customized function to new column    
RFM_Segmentation['Segment_Group'] = RFM_Segmentation.RFM_Score.apply(lambda x: func_customized_segmentation(x))

RFM_Segmentation.head()
df_purshace_contribution = pd.DataFrame(RFM_Segmentation.groupby('Segment_Group')['MonetaryValue'].sum()).reset_index().rename( columns = {'index': 'Segment_Group', 'MonetaryValue': 'Purshace_Contribution'})
df_purshace_contribution
plt.figure(figsize= (10,6))

_ = sns.barplot( data = df_purshace_contribution, x = 'Segment_Group', y = 'Purshace_Contribution', order= ['Gold', 'Silver', 'Bronze'])

total = df_purshace_contribution.Purshace_Contribution.sum()

for p in _.patches:
    height = p.get_height()
    _.text(p.get_x() + p.get_width()/2., height + 90000, '{} USD   -   {}%'.format(round(height,1), round((round(height / total, 3) * 100), 1)), ha="center", va='center', fontsize=10)  # additional round() method to avoid the extra decimal points

_.set_title('Segments Groups', pad = 10, weight= 'bold')
_.set_xlabel('Groups', weight= 'bold')
_.set_ylabel('Purchase Value + Contribution', weight= 'bold')

plt.show()
clean_df.head()
RFM_Segmentation.head()
# Check the unique users in both the RFM_Segmentation and clean_df 
print(len(RFM_Segmentation))
print(len(clean_df.CustomerID.unique()))
arr_customer = np.array(RFM_Segmentation.reset_index().CustomerID)
arr_rfm_data = np.array(RFM_Segmentation[['RFM_Segment', 'RFM_Score', 'Segment_Group']])
rfm_segment_list = []
rfm_score_list = []
seg_group_list = []

# Add weather data into lists 
def match_customers(x):
    try:
        indx = np.where(arr_customer == x)[0][0]
        rfm_segment_list.append(arr_rfm_data[indx][0])
        rfm_score_list.append(arr_rfm_data[indx][1])
        seg_group_list.append(arr_rfm_data[indx][2])
        return 'Transaction Done'
    except:
        rfm_segment_list.append(np.nan)
        rfm_score_list.append(np.nan)
        seg_group_list.append(np.nan)
        return 'Error'

clean_df['RFM_Transaction_Status'] = clean_df.CustomerID.apply(lambda x: match_customers(x))
# Adding lists to new columns 
clean_df['RFM_Segment'] = rfm_segment_list
clean_df['RFM_Score'] = rfm_score_list
clean_df['Group'] = seg_group_list
# Check transactions
clean_df.RFM_Transaction_Status.value_counts()
# Check random reconds
clean_df.loc[clean_df.CustomerID == 12349.0][:1]
# 1. Countries Oviewview

# I break the code into lines to make it more readable
grouped_countries = clean_df.groupby('Country').agg({ 'InvoiceNo': 'count', 'Quantity': 'sum', 'TotalPrice': 'sum' }).round(1)\
                                                                                                                     .sort_values('TotalPrice', ascending = False)\
                                                                                                                     .reset_index()
# Rename InvoiceNo column to TotalTransactions
grouped_countries.rename( columns= {'InvoiceNo': 'TotalTransactions'}, inplace= True)
grouped_countries[: 5]
plt.figure(figsize= (30,8))

_ = sns.barplot( data = grouped_countries,  x = 'Country', y = 'TotalPrice', palette= 'BuGn_r')

_.set_xticklabels(_.get_xticklabels(), rotation=90)
_.set_title('Top countries Based on Total Price (in USD)', pad = 10, weight= 'bold')
_.set_xlabel('Country', weight= 'bold')
_.set_ylabel('Total Price', weight= 'bold')

total = grouped_countries.TotalPrice.sum()

for p in _.patches:
    height = p.get_height()
    _.text(p.get_x() + p.get_width()/2., height + 550000, '{} - {}%'.format(round(height,1), round((round(height / total, 3) * 100), 1)), ha="center", va='center', fontsize=10, rotation = 90)  # additional round() method to avoid the extra decimal points

plt.show()

# NOTE - EIRE is Ireland
# Shortlist the top countries based on revenue (TotalPrice)
grouped_top_countries = grouped_countries.loc[grouped_countries.TotalPrice > 100000]
ordered_totalOrders = grouped_top_countries.sort_values('TotalTransactions', ascending= False)['Country'].tolist()
ordered_totalQuantity = grouped_top_countries.sort_values('Quantity', ascending= False)['Country'].tolist()

grouped_top_countries
# Create annotation function
def annotate_perct(ax_plot, total, add_height):
    
    for p in ax_plot.patches:
        height = p.get_height()
        ax_plot.text(p.get_x() + p.get_width()/2., height + add_height, '{}%'.format(round((round(height / total, 3) * 100), 1)), ha="center", va='center', fontsize=10, rotation = 90)  # additional round() method to avoid the extra decimal points


#  Generate plot for the top countries based on their revenue, transactions, quantity ordered
fig, ax = plt.subplots(ncols = 3, figsize= (30,6))

sns.barplot( data = grouped_top_countries,  x = 'Country', y = 'TotalTransactions', palette= 'BuGn_r', order= ordered_totalOrders,ax= ax[0])
sns.barplot( data = grouped_top_countries,  x = 'Country', y = 'Quantity', palette= 'BuGn_r', order= ordered_totalQuantity, ax= ax[1])
sns.barplot( data = grouped_top_countries,  x = 'Country', y = 'TotalPrice', palette= 'BuGn_r', ax= ax[2])

annotate_perct( ax_plot = ax[0], total = grouped_countries.TotalTransactions.sum(), add_height = 20000)
annotate_perct( ax_plot = ax[1], total = grouped_countries.Quantity.sum(), add_height = 140000)
annotate_perct( ax_plot = ax[2], total = grouped_countries.TotalPrice.sum(), add_height = 260000)

ax[0].set_xticklabels(ax[0].get_xticklabels(), rotation=90)
ax[0].set_title('Top countries Based on Transaction', pad = 10, weight= 'bold')
ax[0].set_xlabel('Country', weight= 'bold')
ax[0].set_ylabel('Transactions', weight= 'bold')

ax[1].set_xticklabels(ax[1].get_xticklabels(), rotation=90)
ax[1].set_title('Top countries Based on Quantity', pad = 10, weight= 'bold')
ax[1].set_xlabel('Country', weight= 'bold')
ax[1].set_ylabel('Quantity', weight= 'bold')

ax[2].set_xticklabels(ax[2].get_xticklabels(), rotation=90)
ax[2].set_title('Top countries Based on Total Price', pad = 10, weight= 'bold')
ax[2].set_xlabel('Country', weight= 'bold')
ax[2].set_ylabel('Total Price', weight= 'bold')

plt.show()
# create another groupby for segmentations / recency/ frecuently, groups and plot them
# 2. Products Overview

# I break the code into lines to make it more readable
grouped_products = clean_df.groupby('Description').agg({ 'InvoiceNo': 'count', 'Quantity': 'sum', 'TotalPrice': 'sum' }).round(1)\
                                                                                                                     .sort_values('TotalPrice', ascending = False)\
                                                                                                                     .reset_index()
# Rename InvoiceNo column to TotalTransactions
grouped_products.rename( columns= {'Description': 'Product', 'InvoiceNo': 'TotalOrders', 'Quantity': 'TotalQuantity'}, inplace= True)
grouped_products[: 5]
# Shortlist the products dataset to the top 25 based on total price 
grouped_products_top25 = grouped_products[:25]
ordered_totalOrders = grouped_products_top25.sort_values('TotalOrders', ascending= False)['Product'].tolist()
ordered_totalQuantity = grouped_products_top25.sort_values('TotalQuantity', ascending= False)['Product'].tolist()
fig, ax = plt.subplots(ncols = 3, figsize= (50,6))

sns.barplot( data = grouped_products_top25[:25],  x = 'Product', y = 'TotalOrders', palette= 'BuGn_r', order= ordered_totalOrders, ax= ax[0])
sns.barplot( data = grouped_products_top25[:25],  x = 'Product', y = 'TotalQuantity', palette= 'BuGn_r', order= ordered_totalQuantity, ax= ax[1])
sns.barplot( data = grouped_products_top25[:25],  x = 'Product', y = 'TotalPrice', palette= 'BuGn_r', ax= ax[2])

ax[0].set_xticklabels(ax[0].get_xticklabels(), rotation=90)
ax[0].set_title('Top countries Based on Transaction', pad = 10, weight= 'bold')
ax[0].set_xlabel('Product', weight= 'bold')
ax[0].set_ylabel('Orders', weight= 'bold')

ax[1].set_xticklabels(ax[1].get_xticklabels(), rotation=90)
ax[1].set_title('Top countries Based on Quantity', pad = 10, weight= 'bold')
ax[1].set_xlabel('Product', weight= 'bold')
ax[1].set_ylabel('Quantity', weight= 'bold')

ax[2].set_xticklabels(ax[2].get_xticklabels(), rotation=90)
ax[2].set_title('Top countries Based on Quantity', pad = 10, weight= 'bold')
ax[2].set_xlabel('Product', weight= 'bold')
ax[2].set_ylabel('Total Price', weight= 'bold')

plt.show()
# 3. Countries Vs Groups Overview

# List of the top countries based on revenue
top_list = grouped_top_countries.Country.tolist()

# I break the code into lines to make it more readable
grouped_countries_vs_groups = clean_df.loc[clean_df.Country.isin(top_list)].groupby(['Country', 'Group']).agg({ 'TotalPrice': 'sum' }).round(1)\
                                                                                                                                      .sort_values('TotalPrice', ascending = False)\
                                                                                                                                      .reset_index()

# Rename InvoiceNo column to TotalTransactions
grouped_countries_vs_groups
# Plot Countries Vs Groups Overview

plt.figure(figsize= (12,6))

_ = sns.barplot( data = grouped_countries_vs_groups,  x = 'Country', y = 'TotalPrice', hue= 'Group')

_.set_xticklabels(_.get_xticklabels(), rotation=90)
_.set_title('Top countries Based on Total Price (in USD) & Customers Groups', pad = 30, weight= 'bold')
_.set_xlabel('Country', weight= 'bold')
_.set_ylabel('Total Price', weight= 'bold')
_.legend(title = 'Groups', loc = 1)

total = grouped_countries_vs_groups.TotalPrice.sum()

for p in _.patches:
    
    # Check if the height is null (as some countries don't have all groups)
    if np.isnan(p.get_height()):
        height = 0
    else:
        height = p.get_height()

    _.text(p.get_x() + p.get_width()/2., height + 250000, '{}%'.format(round((round(height / total, 3) * 100), 1)), ha="center", va='center', fontsize=10, rotation = 90)  # additional round() method to avoid the extra decimal points

plt.show()

# NOTE - EIRE is Ireland
RFM_Segmentation.describe()
# Plotting the actual values

fig, ax = plt.subplots(ncols= 4, figsize= (30,5))

sns.distplot(np.log(RFM_Segmentation.Recency), ax = ax[0])
sns.distplot(np.log(RFM_Segmentation.Frequency), ax = ax[1])
sns.distplot(np.log(RFM_Segmentation.MonetaryValue), ax = ax[2])
sns.distplot(np.log(RFM_Segmentation.RFM_Score), ax = ax[3])

plt.show()
# Transform data
    # 1 - convert int type to float (ensure that you pass numeric variables only to the function)
    # 2 - assign the transformed data into new variable 

# Convert data type function
def  convert_to_float(df, col):

    if df[col].dtype != 'float64' or df[col].dtype != 'float32':
        return df[col].astype('float64')

# Take a copy of the numeric variables
log_transformed_data = RFM_Segmentation.drop(['RFM_Segment', 'Segment_Group'], axis= 1).copy()

# Convert data types
for col in log_transformed_data.columns.tolist():
    log_transformed_data[col] = convert_to_float(log_transformed_data, col)

# Transform the data using np.log
log_transformed_data = np.log(log_transformed_data)
from sklearn.cluster import KMeans 
from sklearn.preprocessing import StandardScaler # To normalize the data by subtract mean and divide by STD. StandardScaler will combine centering and scaling.
# Scale data

scaler = StandardScaler()
scaled_data = scaler.fit_transform(log_transformed_data)
sse = {}

for k in range(1,15):
    
    kmean = KMeans( n_clusters= k, n_jobs= -1 )
    kmean.fit(scaled_data)
    sse[k] = kmean.inertia_
plt.figure(figsize= (10, 7))

_ = sns.pointplot(x= list(sse.keys()), y= list(sse.values()))

_.set_title('Elbow Method', pad = 15, weight= 'bold')
_.set_xlabel('Number of Clusters', weight= 'bold')
_.set_ylabel('Sum of Squared Distances From the Closest Cluster Center', weight= 'bold')

plt.show()
# Best practice to choose the cluster is to choose the point of elbow or the next one.
# In this case, I will choose the 3 clusters.

kmean_3 = KMeans(n_clusters= 3, n_jobs= -1)
kmean_3.fit(scaled_data)

# Extract the cluster's labels
cluster_labels = kmean_3.labels_

# Assign labels to the RFM_Segmentation
RFM_Segmentation = RFM_Segmentation.assign(Clusters = cluster_labels)

RFM_Segmentation.head()
clusters_summary_statistics = RFM_Segmentation.groupby('Clusters').agg({ 'Recency': 'mean',
                                                                         'Frequency': 'mean',
                                                                         'MonetaryValue': ['mean', 'count'] }).round(0)

clusters_summary_statistics
clusters_summary_melt = pd.melt( frame =  RFM_Segmentation.reset_index(), 
                                 id_vars = ['CustomerID','Clusters'], 
                                 value_vars = ['Recency', 'Frequency', 'MonetaryValue'], 
                                 var_name = 'Attribute',
                                 value_name = 'Value')
# Snake Visualization To Understand / Compare Segments for Market Research

plt.figure(figsize= (10, 6))

_ = sns.lineplot( data = clusters_summary_melt, x = 'Attribute', y = 'Value', hue = 'Clusters')

_.set_title('Comparing Segments', pad = 15, weight= 'bold')
_.set_xlabel('Attributes', weight= 'bold')
_.set_ylabel('Values', weight= 'bold')

plt.show()
# To identify relative importance of each attribute
#    1. Get the average value of each cluster
#    2. Get the average of the population
#    3. Get the importance score by dividing clusters averages & population average and subtract 1. 
#    Note - The further the ratio is from the zerop, the more important that attribute is for the segment

cluster_average = RFM_Segmentation.groupby('Clusters')['Recency', 'Frequency', 'MonetaryValue'].mean()
population_average = cluster_average.mean()
relative_importance = cluster_average / population_average - 1
relative_importance.round(2)
plt.figure(figsize= (8, 3))

_ = sns.heatmap( data = relative_importance, annot= True, fmt = '.2f', cmap = 'YlGnBu')

_.set_title('Relative Importance of Attributes', pad = 15, weight= 'bold')

plt.show()
%%timeit

arr_customer = np.array(RFM_Segmentation.reset_index().CustomerID)
arr_clusters = np.array(RFM_Segmentation.reset_index().Clusters)

clusters_list = []

# Add weather data into lists 
def match_customers_cluster(x):
    try:
        indx = np.where(arr_customer == x)[0][0]
        clusters_list.append(arr_clusters[indx])
        return 'Transaction Done'
    except:
        clusters_list.append(np.nan)
        return 'Error'

clean_df['Cluster_Transaction_Status'] = clean_df.CustomerID.apply(lambda x: match_customers_cluster(x))
clean_df['Cluster'] = clusters_list
%%timeit

# Create groupby CustomerID and get the tenure period (max date - min date)
tenure = clean_df.groupby('CustomerID').agg({ 'InvoiceDate': lambda x: (x.dt.date.max() - x.dt.date.min()).days })

arr_customer = np.array(tenure.reset_index().CustomerID)
arr_days = np.array(tenure.reset_index().InvoiceDate)

days_list = []

# Add weather data into lists 
def match_customers_tenure(x):
    try:
        indx = np.where(arr_customer == x)[0][0]
        days_list.append(arr_days[indx])
        return 'Transaction Done'
    except:
        days_list.append(np.nan)
        return 'Error'

clean_df['Tenure_Transaction_Status'] = clean_df.CustomerID.apply(lambda x: match_customers_tenure(x))
clean_df['Tenure'] = days_list
clean_df.head()
# Exporting Dataset to CSV
#clean_df.drop(['RFM_Transaction_Status', 'Cluster_Transaction_Status','Tenure_Transaction_Status'], axis= 1).to_csv('Analyzed_Dataset.csv')
