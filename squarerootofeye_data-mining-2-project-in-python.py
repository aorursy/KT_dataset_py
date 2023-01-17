import bq_helper
import pandas as pd
import os
#from bq_helper import BigQueryHelper

# https://www.kaggle.com/sohier/introduction-to-the-bq-helper-package
# https://github.com/SohierDane/BigQuery_Helper/blob/master/bq_helper.py

# Establish Helper Object for data scanning
google_analytics = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="google_analytics_sample")

# Another example of how to get the data
# bq_assistant = BigQueryHelper("bigquery-public-data", "google_analytics_sample")

# Create list of tables to later assist with queries
tablelist = google_analytics.list_tables()
print(len(tablelist))
print("First table:", tablelist[0],"  Last table:", tablelist[-1])
print("Table ID:", tablelist[0])
print(google_analytics.head(tablelist[0]).columns)
google_analytics.head(tablelist[0], num_rows=3)
print("Table ID:", tablelist[0])
google_analytics.table_schema(tablelist[0])
# Determine Total Datasize
query_everything = """
#standardSQL
SELECT *
FROM 
    # enclose table names with WILDCARDS in backticks `` , not quotes ''
    `bigquery-public-data.google_analytics_sample.ga_sessions_*`
WHERE
    _TABLE_SUFFIX < '20170802'
"""
google_analytics.estimate_query_size(query_everything)
query_oneTable = """
#standardSQL
SELECT *
FROM 
    # enclose table names with WILDCARDS in backticks `` , not quotes ''
    `bigquery-public-data.google_analytics_sample.ga_sessions_20160801`
"""
google_analytics.estimate_query_size(query_oneTable)
oneTable = google_analytics.query_to_pandas_safe(query_oneTable, max_gb_scanned=.1)
oneTable.head(3)
print(oneTable.shape, oneTable.columns)
for col in oneTable.columns:
    print(col, ": ", type(oneTable[col][0]))
    print(oneTable[col][0])
    #print(oneTable[col])
    
# myfilepath = "C:/Users/Jusitn/Documents/Python Scripts/DataMining2"
oneTable.to_csv("20160801.csv", encoding='utf-8', index=False)
import pandas as pd
googledata = pd.read_csv('../input/20160801.csv')
#DOES NOT WORK:  pd.read_csv('../input/data-mining-2-project-in-python/20160801.csv')
googledata.head(1)
for col in googledata.columns:
    print(col, ": ", type(googledata[col][0]))
    print(googledata[col][0])
    #print(oneTable[col])
googledata['trafficSource'][0]
