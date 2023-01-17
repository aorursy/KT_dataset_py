# Print me to document when run:
import datetime
print("Start Run: " + str(datetime.datetime.now()) )
# Imports:
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
# import numpy as np
# import pandas as pd
# import bq_helper
# import csv
print('done')
# Code created by Kaggle when initializing notebook.
# KEPT AS IS:
# ---------------------------------------------------
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
print(os.listdir("../"))

# Any results you write to the current directory are saved as output.
# Load data using BigQueryHelper:
# I'm using some existing code provided by: 
# https://www.kaggle.com/juliaelliott/ga-bigquery-starter-kernel/notebook
# Also found a great article/ help from Megan Risdal:
# https://medium.com/google-cloud/learning-to-analyze-huge-bigquery-datasets-using-python-on-kaggle-2c6c6153f542

import bq_helper
# use the BQHelper library to pull datasets/tables from BigQuery

print("Loading Training Data Set...")
ga_bq_train = bq_helper.BigQueryHelper(active_project= "kaggle-public-datasets", dataset_name = "ga_train_set")
print("Loading Test Data Set... ")
ga_bq_test = bq_helper.BigQueryHelper(active_project= "kaggle-public-datasets", dataset_name = "ga_test_set")
print("Data Loaded: done")
# What is ga_bq_test? 
# For my reference: some of the BigQueryHelper methods
# -------------------------------------------------------
# type(ga_bq_test) # bq_helper.BigQueryHelper
# ga_bq_test.BYTES_PER_GB # 1073741824
# ga_bq_test.tables # {}
# ga_bq_test.client # <google.cloud.bigquery.client.Client at 0x7f1bf9a3ea90>
# ga_bq_test.dataset # Dataset(DatasetReference('kaggle-public-datasets', 'ga_test_set'))
# ga_bq_test.dataset_name # 'ga_test_set'
# ga_bq_test.head # expects the table name head('table_name')
# ga_bq_test.project_name # 'kaggle-public-datasets'
# ga_bq_test.total_gb_used_net_cache # 0
# ga_bq_test.list_tables() # a list of all the tables
# ga_bq_test.table_schema(table_name) # schema of a specific table
# First look at what we have to work with:
print( "Number of Tables in Train Data Set: " + str(len(ga_bq_train.list_tables())) )
print( "Number of Tables in Test Data Set: " + str(len(ga_bq_test.list_tables())) )
ga_bq_test_list_tables = ga_bq_test.list_tables()
ga_bq_test_list_tables[0:3]
#
# Number of Tables in Train Data Set: 366
# Number of Tables in Test Data Set: 272
# ['ga_sessions_20170802',
#  'ga_sessions_20170803',
# ...
# OUTPUT a csv file of the list of tables
import csv

# TRAIN : ga_bq_train
print('saving... ga_bq_train_list_tables ...')
list_of_tables = ga_bq_train.list_tables()
csvfile = "ga_bq_train_list_tables.csv"
#Assuming res is a flat list
with open(csvfile, "w") as output:
    writer = csv.writer(output, lineterminator='\n')
    for table in list_of_tables:
        writer.writerow([table])    
print('done')

# TEST : ga_bq_test
print('saving... ga_bq_test_list_tables ...')
list_of_tables = ga_bq_test.list_tables()
csvfile = "ga_bq_test_list_tables.csv"
#Assuming res is a flat list
with open(csvfile, "w") as output:
    writer = csv.writer(output, lineterminator='\n')
    for table in list_of_tables:
        writer.writerow([table])    
print('done')
# TABLE AND SCHEMA INSPECTION VIA GOOGLE SHEETS
# I used the above csv file to inpsect data in Google Sheets :
# https://docs.google.com/spreadsheets/d/1XBRL5cmJwqGe2KENLITTL_4DY-DOeAr4-JzlUYoMdYw/edit?usp=sharing
#
# NOTES:
# -- The main field to test against will be: totals.transactionRevenue
# -- A number of fields will NOT be needed
# -- The 'time' seems to NOT include timezone
# TEST table schema:
# Take a look at the TEST set schema for one of the tables
print(ga_bq_test.table_schema('ga_sessions_20170802').shape)
ga_bq_test_table_schema_ga_sessions_20170802 = ga_bq_test.table_schema('ga_sessions_20170802')
ga_bq_test_table_schema_ga_sessions_20170802.head(5)
# There are 185 'columns'
# TRAIN table schema:
# Take a look at the TRAIN set schema for one of the tables
print(ga_bq_train.table_schema('ga_sessions_20160801').shape)
ga_bq_train_table_schema_ga_sessions_20160801 = ga_bq_train.table_schema('ga_sessions_20160801')
ga_bq_train_table_schema_ga_sessions_20160801.head(5)
# There are 185 'columns'
# Save the schema outputs to csv files:
# a great quick reference: https://www.kaggle.com/szamil/where-is-my-output-file
print('saving csv file...')
ga_bq_test_table_schema_ga_sessions_20170802.to_csv('ga_bq_test_table_schema_ga_sessions_20170802.csv', index = False)
ga_bq_train_table_schema_ga_sessions_20160801.to_csv('ga_bq_train_table_schema_ga_sessions_20160801.csv', index = False)
print('done')
# Just checking the way to reference specific column name:
reference_schema = ga_bq_test.table_schema('ga_sessions_20170802')
print( reference_schema.name[184] )
reference_schema.name[184]
# I want to make sure all of the tables have the same schema:
# for an item in list I want to look up the schema and compare it to a reference schema
# I'm just picking a field at random, and comparing the dimension size
#
# ----------- commentted out no need to re-run each commit -----------
# ----------- all tables do appear to have same schema -----------
'''
# CHECK TEST SET:
# let's use the first table within the test set of data as our reference:
reference_schema = ga_bq_test.table_schema('ga_sessions_20170802')
reference_shape = reference_schema.shape
print(reference_shape)
#
print('checking test data schema')
ga_bq_test_list_tables = ga_bq_test.list_tables()
print(len(ga_bq_test_list_tables))
# keep a count of the table we are on for easy tracking (old school coding)
nnn = 0
for table in ga_bq_test_list_tables:
    nnn = nnn + 1
    # grab the schema
    table_to_check_schema = ga_bq_test.table_schema(table)
    table_to_check_shape = table_to_check_schema.shape
    # ok, I'm going to cheat:
    # I'm going to check if the shapes are the same
    # and I'll spot check one random column near the end.
    if (reference_shape == table_to_check_shape) and (table_to_check_schema.name[150] == reference_schema.name[150]):
        print("table: " + str(nnn) + " : " + table + " check ok" )
    else:
        print("table: " + str(nnn) + " : " + table + " check NOT OK")
        break
print('TEST set done')
print(' ')

# CHECK TRAIN SET:
# let's use the first table within the train set of data as our reference:
reference_schema = ga_bq_train.table_schema('ga_sessions_20160801')
reference_shape = reference_schema.shape
print(reference_shape)
#
print('checking train data schema')
ga_bq_train_list_tables = ga_bq_train.list_tables()
print(len(ga_bq_train_list_tables))
nnn = 0
for table in ga_bq_train_list_tables:
    nnn = nnn + 1
    # grab the schema
    table_to_check_schema = ga_bq_train.table_schema(table)
    table_to_check_shape = table_to_check_schema.shape
    # ok, I'm going to cheat:
    # I'm going to check if the shapes are the same
    # and I'll spot check one random column near the end.
    if (reference_shape == table_to_check_shape) and (table_to_check_schema.name[150] == reference_schema.name[150]):
        print("table: " + str(nnn) + " : " + table + " check ok" )
    else:
        print("table: " + str(nnn) + " : " + table + " check NOT OK")
        #print(table_to_check_shape)
        #print(table_to_check_schema.name[150])
        #print(reference_schema.name[150])
        break
print('TRAIN set done')
'''
# ----------- end of commented section -----------
#
# Results:
# OK, 
# I commented this out because it takes some time, and I only needed to do it once to convince myself
# 
# The Test data and Train data sets are the same except:
# The Train set has one additional totals.transactionRevenue column
#
print('done')
# DATA FIELDS AS FROM THE CONTEST DESCRIPTION
# https://www.kaggle.com/c/ga-customer-revenue-prediction/data
#
# Data Fields:
# fullVisitorId- A unique identifier for each user of the Google Merchandise Store.
# channelGrouping - The channel via which the user came to the Store.
# date - The date on which the user visited the Store.
# device - The specifications for the device used to access the Store.
# geoNetwork - This section contains information about the geography of the user.
# sessionId - A unique identifier for this visit to the store.
# socialEngagementType - Engagement type, either "Socially Engaged" or "Not Socially Engaged".
# totals - This section contains aggregate values across the session.
# trafficSource - This section contains information about the Traffic Source from which the session originated.
# visitId - An identifier for this session. This is part of the value usually stored as the _utmb cookie. This is only unique to the user. For a completely unique ID, you should use a combination of fullVisitorId and visitId.
# visitNumber - The session number for this user. If this is the first session, then this is set to 1.
# visitStartTime - The timestamp (expressed as POSIX time).
# A query estimate for a single table within the training dataset:
# ga_sessions_20160801
queryy = \
    """
    SELECT  *
    FROM `kaggle-public-datasets.ga_train_set.ga_sessions_20160801` 
    """

print("size = " + str( ga_bq_train.estimate_query_size(queryy) * 1000 ) + " MB" )
# A query for a single table within the training dataset
# ga_sessions_20160801
queryy = \
    """
    SELECT  *
    FROM `kaggle-public-datasets.ga_train_set.ga_sessions_20160801`
    """

print('a_small_query: start...')
ga_train_set_ga_sessions_20170803 = ga_bq_train.query_to_pandas_safe(queryy)
print('a_small_query: done.')

ga_train_set_ga_sessions_20170803.describe()
ga_train_set_ga_sessions_20170803.info()
ga_train_set_ga_sessions_20170803.shape
print('saving csv file...')
ga_train_set_ga_sessions_20170803.to_csv('ga_train_set_ga_sessions_20170803.csv', index = False)
print('done')
# SPECIFIC TABLE INSPECTION VIA GOOGLE SHEET
# I used the above csv file to inpsect a single table within Google Sheets :
# https://docs.google.com/spreadsheets/d/1-kn-O6H4p_jNbfF0cN2i5o-FG1AN2leBTdXAzFDDdeQ/edit?usp=sharing
# A query estimate for a ALL tables within the training dataset:
queryy = \
    """
    SELECT  *
    FROM `kaggle-public-datasets.ga_train_set.ga_sessions_*` 
    """

print("size = " + str( ga_bq_train.estimate_query_size(queryy) * 1000 ) + " MB" )
# size = 737.8220958635211 MB
# query the BQ train set tables to summarize total transaction revenue per user, where fullVisitorId is unique per user.
queryy = \
    """
    SELECT  fullVisitorId, coalesce(SUM( totals.transactionRevenue ),0) AS total_transactionrevenue_per_user
    FROM `kaggle-public-datasets.ga_train_set.ga_sessions_*` 
    GROUP BY fullVisitorId
    ORDER BY total_transactionrevenue_per_user DESC
    """

print("size = " + str( ga_bq_train.estimate_query_size(queryy) * 1000 ) + " MB" )
# ok, let's do the query and save it:
print('starting queryy...')
total_revenue_per_unique_fullVisitorID = ga_bq_train.query_to_pandas_safe(queryy)
print('done')
# This took some time to run
type(total_revenue_per_unique_fullVisitorID)
#Take a look at some basic analytics of the transaction revenue pulled from our query
total_revenue_per_unique_fullVisitorID.describe()
total_revenue_per_unique_fullVisitorID.head(5)
# Save output.
print('saving csv file...')
total_revenue_per_unique_fullVisitorID.to_csv('total_revenue_per_unique_fullVisitorID.csv', index = False)
print('done')
#How many of the users spent any money?
len(total_revenue_per_unique_fullVisitorID[total_revenue_per_unique_fullVisitorID.total_transactionrevenue_per_user > 0])
# 9996
total_revenue_per_unique_fullVisitorID.shape
# (714167, 2)
#Let's take a look at the average transaction revenue for each user who spent money.

queryy= \
"""
SELECT
( (SUM(total_transactionrevenue_per_visitorid) / SUM(total_visits_per_visitorid))/1000000 ) AS avg_revenue_by_user_per_visit_as_dollars
FROM (
    SELECT
    fullVisitorId,
    SUM( totals.visits ) AS total_visits_per_visitorid,
    SUM( totals.transactionRevenue ) AS total_transactionrevenue_per_visitorid
    FROM
    `kaggle-public-datasets.ga_train_set.ga_sessions_*`
    WHERE
    totals.visits > 0
    AND totals.transactionRevenue IS NOT NULL
    GROUP BY
    fullVisitorId );
"""

avg_transrev = ga_bq_train.query_to_pandas_safe(queryy, max_gb_scanned=10)
print('done')
avg_transrev.head(5)
# $133.75
# another query to pull a few specific datapoints per user.
print('starting query...')
queryy = \
"""
SELECT  fullVisitorId, 
    SUM(totals.transactionRevenue) AS total_transactionrevenue_per_user, 
    SUM(totals.pageviews) AS total_pagesviews_per_user,
    SUM(totals.visits ) AS total_visits_per_user,
    SUM(totals.timeOnSite ) AS total_timeonsite_per_user
FROM `kaggle-public-datasets.ga_train_set.ga_sessions_*` 
GROUP BY fullVisitorId
ORDER BY total_transactionrevenue_per_user DESC
"""
all_train_summary2 = ga_bq_train.query_to_pandas_safe(queryy)
print('done')
all_train_summary2.head()
#Now let's drop the data from that query into a dataframe.
#This is the start of how you might pull features that you've queried out of BQ to begin modelling.
full_df = ga_bq_train.query_to_pandas(queryy)
full_df.describe()
# END OF REFERENCE FROM 
# https://www.kaggle.com/juliaelliott/ga-bigquery-starter-kernel/notebook
# JULIA ELLIOTT
# Thanks Julia!
# ok... what's next...
# This, is all of the users
# fullVisitorID = 4984366501121503466
queryy = \
    """
    SELECT  
    fullVisitorId, 
    sessionId,
    date,
    visitStartTime,
    coalesce(totals.transactionRevenue,0) AS totals_transactionRevenue
    FROM `kaggle-public-datasets.ga_train_set.ga_sessions_*` 
    ORDER BY visitStartTime
    """
print("size = " + str( ga_bq_train.estimate_query_size(queryy) * 1000 ) + " MB" )
print('starting query...')
allUserQuery_event_date_revenue = ga_bq_train.query_to_pandas_safe(queryy)
print('done')
allUserQuery_event_date_revenue.head(5)
allUserQuery_event_date_revenue.describe()
# Save output.
print('saving csv file...')
allUserQuery_event_date_revenue.to_csv('allUserQuery_event_date_revenue.csv', index = False)
print('done')
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
figure(num=None, figsize=(15, 4), dpi=80, facecolor='w', edgecolor='k')
x = allUserQuery_event_date_revenue.visitStartTime
y = allUserQuery_event_date_revenue.totals_transactionRevenue
plt.plot(x, y, 'o', color='black');
# This, is all of the users
# fullVisitorID = 4984366501121503466
queryy = \
    """
    SELECT  
    fullVisitorId, 
    sessionId,
    date,
    visitStartTime,
    coalesce(totals.transactionRevenue,0) AS totals_transactionRevenue
    FROM `kaggle-public-datasets.ga_train_set.ga_sessions_*` 
    WHERE fullVisitorId = '4984366501121503466'
    ORDER BY visitStartTime
    """
print("size = " + str( ga_bq_train.estimate_query_size(queryy) * 1000 ) + " MB" )
print('start query')
userData_4984366501121503466 = ga_bq_train.query_to_pandas_safe(queryy)
print('done')
userData_4984366501121503466.head(5)
# Let's save it:
print('saving csv file...')
userData_4984366501121503466.to_csv('userData_4984366501121503466.csv', index = False)
print('done')
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
figure(num=None, figsize=(15, 4), dpi=80, facecolor='w', edgecolor='k')
x = userData_4984366501121503466.visitStartTime
y = userData_4984366501121503466.totals_transactionRevenue
markerline, stemlines, baseline = plt.stem(x, y, '-.')
# setting property of baseline with color red and linewidth 2
plt.setp(baseline, color='r', linewidth=2)
plt.show()
#
# Interesting points about this specific user:
# Most visits result in revenue
# Is there a trend, or re-occuring pattern?
# will need to lay seasons or events along time path.

# just checking how to sort by time using the pandas sort_values() method:
allUserQuery_event_date_revenue.sort_values(by=['visitStartTime'])
allUserQuery_event_date_revenue.head(5)
# I want the SUM for each individual user
queryy = \
    """
    SELECT  
    fullVisitorId, 
    MIN(date) AS minimum_date,
    MIN(visitStartTime) AS minimum_vistStartTime,
    coalesce(SUM( totals.transactionRevenue ),0) AS total_transactionrevenue_per_user
    FROM `kaggle-public-datasets.ga_train_set.ga_sessions_*` 
    GROUP BY fullVisitorId
    ORDER BY total_transactionrevenue_per_user DESC
    """
print("size = " + str( ga_bq_train.estimate_query_size(queryy) * 1000 ) + " MB" )
print('starting query...')
train_sum_for_each_user = ga_bq_train.query_to_pandas_safe(queryy)
print('done')
train_sum_for_each_user.head(5)
# Double checking the data type, (Want to make sure we are working with numbers, not text)
print('minimum_date:')
print(train_sum_for_each_user['minimum_date'].describe())
print('-----')
print('minimum_vistStartTime:')
print(train_sum_for_each_user['minimum_vistStartTime'].describe())
print('-----')
print('total_transactionrevenue_per_user:')
print(train_sum_for_each_user['total_transactionrevenue_per_user'].describe())
# minimum_date is an object (probably text)
# minimum_visitStartTime is float64
# total_transactionrevenue_per_user is float64
# In this silly model,
# I want to calculate the average spend per day,
# for the date range, I want to use the date of the first purchase in the data, divided by the last day of the data.
#
# We are subtracting midnight on 8/1/2017 plus 1.5 days (to make sure we don't get any negitive time, found by trial and error)
# 8/1/2017 is 1501545600 in POSIX time (the time in seconds from 1/1/1970)
#
# This pandas dataframe will be a time delta in seconds:
train_user_time_delta = (1501545600+(1.5*24*60*60)) - train_sum_for_each_user['minimum_vistStartTime'] 
train_user_time_delta.head(5)
train_user_time_delta.describe()
# Now calculating the revenue per day in given units of 1/10^6 dollars per day.
train_user_revenue_per_day = train_sum_for_each_user['total_transactionrevenue_per_user'] / (train_user_time_delta / (60 * 60 * 24) )
train_user_revenue_per_day.head(5)
train_user_revenue_per_day.describe()
# add results to dataframe as new column:
train_sum_for_each_user['revenuePerDay'] = train_user_revenue_per_day
train_sum_for_each_user.head(5)
# checking some stats:
print("count of NaN: " + str( train_sum_for_each_user['revenuePerDay'].isna().sum()) )
print("count of Null: " + str( train_sum_for_each_user['revenuePerDay'].isnull().sum()) )
print("count of not NaN: " + str( train_sum_for_each_user['revenuePerDay'].count()) )
print("rows and columns of data: " + str(train_sum_for_each_user.shape) )
#
# train_sum_for_each_user[train_sum_for_each_user['revenuePerDay'] > 0].count()
print(sum(train_sum_for_each_user['revenuePerDay'] > 0)) # 9996 with revenue
print(sum(train_sum_for_each_user['revenuePerDay'] == 0)) # 704,171 no revenue
print(sum(train_sum_for_each_user['revenuePerDay'] < 0)) # just being complete
# Great!
# now I just need to grab the 'test' data, and apply the silly formula
# time detla of test set * revenue per day for each user = answer!
# let's query for a list of all the unique test users:
# I want the SUM for each individual user
queryy = \
    """
    SELECT  
    fullVisitorId
    FROM `kaggle-public-datasets.ga_test_set.ga_sessions_*` 
    GROUP BY fullVisitorId
    ORDER BY fullVisitorId
    """
print("size = " + str( ga_bq_test.estimate_query_size(queryy) * 1000 ) + " MB" )
print('starting query...')
test_only_unique_user_list = ga_bq_test.query_to_pandas_safe(queryy)
print('done.')
test_only_unique_user_list.head(5)
# Ok... let's create a dataframe where we have the unique list of test users
# I'll be doing a left-join so I can keep all Test users, but match up any Train users:
df_a = test_only_unique_user_list
df_b = train_sum_for_each_user
df_new = pd.merge(df_a, df_b, on='fullVisitorId', how='left')
df_new.head(5)
df_new.describe()
# Im going to do an outer and inner join just to see how things add up:
df_outer = pd.merge(df_a, df_b, on='fullVisitorId', how='outer')
df_inner = pd.merge(df_a, df_b, on='fullVisitorId', how='inner')
print('done')
print('train set unique count: ' + str(df_b.shape)) # (714167, 5)
print('test set unique count: ' + str(df_a.shape)) # (617242, 1)
print('train and test total combined: ' + str(df_outer.shape)) # (1323730, 5)
print('train within test: ' + str(df_inner.shape)) # (7679, 5)
print('count of how many users in test set have postive revenue: ' + str(sum(df_new['revenuePerDay'] > 0)) ) #536
# NOTES: 
# Yes it all adds up: 714,167 + 617,242 - 7,679 = 1,323,730 (Joins work as they should)
# We only have 7,679 matching user IDs between test and train.
# we had 9,996 within Train that had greater than zero revenue (calculated previously)
# in the new Test set caculated with this silly model: we have 536 users with positive revenue
# (assuming that our population would be a similar percentage, we are wayyy under)
print("count of NaN: " + str( df_new['revenuePerDay'].isna().sum()) )
print("count of not NaN: " + str( df_new['revenuePerDay'].count()) )
print("rows and columns of data: " + str(df_new.shape) )
print("percent of data with NaN: " + str(609563/617242*100) + "%")
print("check that sums add up: " + str(609563+7679) + " ... yes, they add up")
# ok, this model sucks
# 98.8% of users in the test set do NOT have data in the train set
df_new.revenuePerDay.head(5)
# getting rid of NaN's to make math work
df_new.revenuePerDay.fillna(0,inplace=True)
df_new.revenuePerDay.head(5)
# I need to simply multiply the revenuePerday by the number of days in the test set for each user
output = np.log(df_new.revenuePerDay * 271+1)
output.head(5)
# finally, let's put it all together:
visit_id = df_new.fullVisitorId
out_PredictedLogRevenue = pd.concat([visit_id, output], axis=1)
out_PredictedLogRevenue.rename(columns={'revenuePerDay':'PredictedLogRevenue'}, inplace=True)
out_PredictedLogRevenue.head(5)
# Let's save it:
print('saving csv file...')
out_PredictedLogRevenue.to_csv('submission_SteveBlack_v001_date_2018_1003_1137.csv', index = False)
print('done')