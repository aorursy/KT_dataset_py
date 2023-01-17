import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
color = sns.color_palette()
plt.style.use('bmh')
plt.set_cmap('spring')
%matplotlib inline
import bq_helper

#Here's how we can use the BQHelper library to pull datasets/tables from BigQuery
ga_bq_train = bq_helper.BigQueryHelper(active_project= "kaggle-public-datasets", 
                                       dataset_name = "ga_train_set")
ga_bq_test = bq_helper.BigQueryHelper(active_project= "kaggle-public-datasets", 
                                       dataset_name = "ga_test_set")

ga_bq_train.list_tables()[:10]
#columns in train dataset
ga_bq_train.table_schema((ga_bq_train.list_tables()[0]))['name'].tolist()
#Let's check the size of train dataset total number of records
total_train_query = """SELECT  COUNT(*) AS COUNT
  FROM `kaggle-public-datasets.ga_train_set.ga_sessions_*` """
total_train = ga_bq_train.query_to_pandas(total_train_query)
print('Total number of records in training dataset :',total_train['COUNT'][0])

#Let's check the size of test dataset total number of records
total_test_query = """SELECT  COUNT(*) AS COUNT
  FROM `kaggle-public-datasets.ga_test_set.ga_sessions_*` """
total_test = ga_bq_test.query_to_pandas(total_test_query)
print('Total number of records in test dataset :',total_test['COUNT'][0])

#training data snapshot
ga_bq_train.head(ga_bq_train.list_tables()[0]) #this will show data in first table of bigquery dataset

#exploration of Target Variable using BigQuery

totalrevenue_per_user_query = """SELECT  fullVisitorId, coalesce(SUM( totals.transactionRevenue ),0) AS totalrevenue_per_user
  FROM `kaggle-public-datasets.ga_train_set.ga_sessions_*` 
  GROUP BY fullVisitorId
"""
totalrevenue_per_user = ga_bq_train.query_to_pandas_safe(totalrevenue_per_user_query)
#plot distribution of transactionRevenue
plt.figure(figsize=(8,6))
#scatter plot on natural log of totalrevenue per user
#original code by : SRK kernel(https://www.kaggle.com/sudalairajkumar/simple-exploration-baseline-ga-customer-revenue)

plt.scatter(range(totalrevenue_per_user.shape[0]), np.sort(np.log1p(totalrevenue_per_user["totalrevenue_per_user"].values)))
plt.xlabel('index', fontsize=12)
plt.ylabel('totalRevenue', fontsize=12)
plt.title('Distribution of totalrevenue per user')
plt.show()
traindate_query = """SELECT MIN(date) as startdate,MAX(date) as enddate 
    FROM `kaggle-public-datasets.ga_train_set.ga_sessions_*`"""
traindate_result =  ga_bq_train.query_to_pandas_safe(traindate_query)
print('Data in training set is for date',traindate_result['startdate'].iloc[0],'to',traindate_result['enddate'].iloc[0])
testdate_query = """SELECT MIN(date) as startdate,MAX(date) as enddate 
    FROM `kaggle-public-datasets.ga_test_set.ga_sessions_*`"""
testdate_result =  ga_bq_test.query_to_pandas_safe(testdate_query)
print('Data in test set is for date',testdate_result['startdate'].iloc[0],'to',testdate_result['enddate'].iloc[0])
revenue_per_date_query = """SELECT  PARSE_DATE('%Y%m%d',date) AS DATE,COUNT(*) AS VISIT_COUNT ,coalesce(SUM( totals.transactionRevenue ),0) AS totalrevenue,
coalesce(AVG( totals.transactionRevenue ),0) AS avgrevenue
  FROM `kaggle-public-datasets.ga_train_set.ga_sessions_*` 
  GROUP BY date
"""
revenue_per_date = ga_bq_train.query_to_pandas_safe(revenue_per_date_query)

for i,col in enumerate(['VISIT_COUNT','totalrevenue','avgrevenue']):
    #fig,axes = plt.subplots(3,1)
    revenue_per_date.plot(x='DATE',y=col,figsize=(8,6))
    if col=='VISIT_COUNT' :
        plt.title('Visits count per day')
    else :
        plt.title('Distribution of ' + col + ' per date')
    plt.xlabel('DATE', fontsize=12)
    plt.ylabel(col, fontsize=12)
    
def categorical_countplot(feature):
    #this function extract usage count of feature passed using BigQuery and visualize the usage of top 10 feature values based on their counts
    separate_feat = feature.split('.')[1]
    query = """SELECT """ + feature + """, COUNT(*) AS COUNT,coalesce(SUM( totals.transactionRevenue ),0) AS TotalRevenue,
    coalesce(AVG( totals.transactionRevenue ),0) AS AvgRevenue
      FROM `kaggle-public-datasets.ga_train_set.ga_sessions_*` 
      GROUP BY """ + feature + """
      ORDER BY COUNT(*) DESC"""
    feature_count = ga_bq_train.query_to_pandas_safe(query)
    print('Total number of ' ,separate_feat, ' :',len(feature_count[separate_feat]))
    #let's visualize the usage of top 10 feature categories using barplot
    plt.figure(figsize=(16,6))
    for i,col in enumerate(['COUNT','TotalRevenue','AvgRevenue']) :
        ax = plt.subplot(1,3,i+1)
        sns.barplot(x=separate_feat,y=col,data=feature_count.head(10))
        if col=='COUNT' :
            plt.title('Visits count per '  + separate_feat)
        else :
            plt.title(col + ' per ' + separate_feat)
        plt.xticks(rotation=90)
# exploration of browser variable
categorical_countplot('device.browser')

# exploration of operating system
categorical_countplot('device.operatingSystem')
categorical_countplot('device.deviceCategory')
categorical_countplot('geoNetwork.continent')
categorical_countplot('geoNetwork.subContinent')
categorical_countplot('geoNetwork.country')
categorical_countplot('trafficSource.source')
categorical_countplot('trafficSource.medium')
hits_query = """SELECT totals.hits as hits, COUNT(*) AS COUNT,coalesce(SUM( totals.transactionRevenue ),0) AS TotalRevenue,
coalesce(AVG( totals.transactionRevenue ),0) AS AvgRevenue
  FROM `kaggle-public-datasets.ga_train_set.ga_sessions_*` 
  GROUP BY totals.hits ORDER BY totals.hits"""

hits_count = ga_bq_train.query_to_pandas_safe(hits_query)
#visits per hit
plt.figure(figsize=(16,8))
sns.barplot(x='hits',y='COUNT',data=hits_count.head(50),color='green')
plt.title('Visits count per hits')
#effect of hits on total revenue and mean revenue
plt.figure(figsize=(16,8))
for i,col in enumerate(['TotalRevenue','AvgRevenue']) :
        ax = plt.subplot(1,2,i+1)
        sns.scatterplot(x='hits',y=col,data=hits_count)
        plt.title(col + ' per hits')
        scale_y = 1e6
        ticks_y = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x/scale_y))
        ax.yaxis.set_major_formatter(ticks_y)
        plt.xticks(rotation=90)
pageviews_query = """SELECT CAST(totals.pageviews as INT64) as pageviews, COUNT(*) AS COUNT,coalesce(SUM( totals.transactionRevenue ),0) AS TotalRevenue,
coalesce(AVG( totals.transactionRevenue ),0) AS AvgRevenue
  FROM `kaggle-public-datasets.ga_train_set.ga_sessions_*` 
  GROUP BY totals.pageviews ORDER BY totals.pageviews"""

pageviews_count = ga_bq_train.query_to_pandas_safe(pageviews_query)
#pageviews counts
plt.figure(figsize=(16,8))
sns.barplot(x='pageviews',y='COUNT',data=pageviews_count.head(30),color='blue')
plt.title('Visit count per pageviews')
#effect of pageviews on total revenue and mean revenue
plt.figure(figsize=(16,8))
for i,col in enumerate(['TotalRevenue','AvgRevenue']) :
        ax = plt.subplot(1,2,i+1)
        sns.scatterplot(x='pageviews',y=col,data=pageviews_count)
        plt.title(col + ' per pageviews')
        scale_y = 1e6
        ticks_y = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x/scale_y))
        ax.yaxis.set_major_formatter(ticks_y)
        plt.xticks(rotation=90)