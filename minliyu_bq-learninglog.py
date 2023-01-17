# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# Load BigQuery packages

from google.cloud import bigquery
# 1. Load the dataset 

client = bigquery.Client()

dataset_ref = client.dataset('hacker_news', project='bigquery-public-data')

dataset = client.get_dataset(dataset_ref)
# 2. List the table_id in loaded datast

client.list_tables(dataset)

# for x in client.list_tables(dataset):

#     print (x.table_id)

[x.table_id for x in client.list_tables(dataset)]
# 3. View the table 'full'

table_ref = dataset.table('full')

full = client.get_table(table_ref)

# 4. Explore available commands(attributes) to the table

print([command for command in dir(full) if not command.startswith('_')])
# 5. Select the schema

fields = [col for col in full.schema if col.name in ['by','title','time']]

print(fields)

full.schema
# 6. Show a slice by the selected schema

client.list_rows(full,max_results=5, start_index=1,selected_fields= fields).to_dataframe()

# 7. Another way to show a slice by the selected schema

result=[x for x in (client.list_rows(full,max_results=5, start_index=500,selected_fields= fields))]

# print([x for x in result])

for i in result:

    print(dict(i))
#8 Incestigate the resaurces would have consumed for a full scan

bytes_per_GB = 2**30 # (Gigibyte) for binary system/note that 10**9 is giga bytes

full.num_bytes/bytes_per_GB
# 9. A helper func to calculate the scanned bytes

def estimate_scanned(query,bq_client):

    dry_run = bigquery.job.QueryJobConfig(dry_run = True)

    job = bq_client.query(query,job_config = dry_run)

    return job.total_bytes_processed/2**30

# 10. The query

my_query = """

           SELECT id FROM `bigquery-public-data.hacker_news.full`

           """

# 11 run the func

estimate_scanned(my_query,client)
# 12 Also the orient way to estimate

dry_run = bigquery.job.QueryJobConfig(dry_run = True) #or use the attribute form-> dry_run = bigquery.job.QueryJob_Config() ->dry_run.dry_run = True

safe_run = bigquery.job.QueryJobConfig(maximum_bytes_billed=2**30)

print(safe_run._properties) # ->1GB

scanned = client.query(my_query,job_config = dry_run)

scanned.total_bytes_processed/2**30

# estimate_scanned (my_query,client)
# dir(dry_run)

test = bigquery.job.QueryJobConfig()

test.dry_run = True

dir(test)

test._properties
import pandas as pd

from bq_helper import BigQueryHelper
# 1. Load the project/dataset

bq_assistant = BigQueryHelper("bigquery-public-data",'openaq') # original coeds: set ref->get_dataset/table

# 2. List all table

# dir(ba_assistant)

bq_assistant.list_tables() # client.list_tables(dataset)-> [x for x in y]
# 3. A quick way as list_rows

bq_assistant.head('global_air_quality',num_rows=5) # client.list_rows(dataset, max_results= 5)
# 4. Table Schema

bq_assistant.table_schema('global_air_quality')
# 5. Write query

QUERY = "SELECT location, timestamp, pollutant FROM `bigquery-public-data.openaq.global_air_quality`"

# df = bq_assistant.query_to_pandas(QUERY)

# df = bq_assistant.query_to_pandas_safe(QUERY)

df = bq_assistant.query_to_pandas_safe(QUERY, max_gb_scanned=.5*10**6)

df.head()
# SOme testing

# dir(bq_assistant)

# bq_assistant._BigQueryHelper__dataset_ref
# 1. Load the Project and dataset

# 1.a Old way:

client = bigquery.Client()

dataset_ref = client.dataset('google_analytics_sample',project='bigquery-public-data' )

dataset = client.get_dataset(dataset_ref)

table_ref = dataset.table('ga_sessions_20170701')

table = client.get_table(table_ref)

# table.schema

print([x for x in table.schema if x.name == 'visitorId'])

# print([x for x in table.schema])



# 1.b New way:from bq_helper import BigQueryHelper

bq_assistant = BigQueryHelper("bigquery-public-data", "google_analytics_sample")

# bq_assistant.list_tables()
# 2. bq_assistant.list_tables()

bq_assistant.head('ga_sessions_20160801',num_rows=5)
# 3. Try query

query = """

        SELECT visitId,totals

        FROM `bigquery-public-data.google_analytics_sample.ga_sessions_20160801`

        """

print('resource consumped in this query is {} Gb'.format(bq_assistant.estimate_query_size(query)))

df = bq_assistant.query_to_pandas_safe(query)

df.head()
#3. See the table schema

# bq_assistant.table_schema('ga_sessions_20160801')

bq_assistant.head('ga_sessions_20160801',num_rows=5)
# Q1: What is the total number of transactions generated per device browser in July 2017?

# bq_assistant.head('ga_sessions_20160801',num_rows=5)

query = """

        SELECT device.browser, sum(totals.transactions) AS num_transactions

        FROM `bigquery-public-data.google_analytics_sample.ga_sessions_*`

        WHERE _TABLE_SUFFIX BETWEEN '20170701' AND '20170731'

        GROUP BY device.browser

        ORDER BY num_transactions DESC

        """

# bq_assistant.estimate_query_size(query)

response1 = bq_assistant.query_to_pandas_safe(query)

response1.head(10)
# Q2: The real bounce rate is defined as the percentage of visits with a single pageview. What was the real bounce rate per traffic source?

query = """

        WITH s AS(

        SELECT trafficSource.source AS source, sum(totals.bounces) AS num_bounces, COUNT(*) AS total_visit

        FROM `bigquery-public-data.google_analytics_sample.ga_sessions_*`

        WHERE _TABLE_SUFFIX BETWEEN '20170701' AND '20170731'

        GROUP BY source)

        SELECT source, num_bounces, total_visit, num_bounces/total_visit*100 AS bounce_rate 

        FROM s

        ORDER BY total_visit DESC

        """

response2=bq_assistant.query_to_pandas_safe(query)

response2.head(10)
# Q3:What was the average number of product pageviews for users who made a purchase in July 2017?

query = """

        WITH s AS (

        SELECT totals.pageviews AS pageviews,totals.transactions AS num_transactions

        FROM `bigquery-public-data.google_analytics_sample.ga_sessions_*`

        WHERE _TABLE_SUFFIX BETWEEN '20170701' AND '20170731'

        AND totals.transactions > 0

        AND totals.transactions IS NOT null

        )

        SELECT SUM(pageviews)/SUM(num_transactions) AS viewPpurchase

        FROM s

        """

bq_assistant.query_to_pandas_safe(query)
# Q4:What was the average number of product pageviews for users who did not make a purchase in July 2017?

query = """

        WITH s AS (

        SELECT totals.pageviews AS pageviews,totals.transactions AS num_transactions

        FROM `bigquery-public-data.google_analytics_sample.ga_sessions_*`

        WHERE _TABLE_SUFFIX BETWEEN '20170701' AND '20170731'

        AND totals.transactions IS null

        )

        SELECT Avg(pageviews) AS AVG_Pageviews

        FROM s

        """

df=bq_assistant.query_to_pandas_safe(query)

df.head(5)
#Q5: What was the average total transactions per user that made a purchase in July 2017?



query = """

        WITH s AS (SELECT fullVisitorId,SUM(totals.transactions) AS num_transactions

        FROM `bigquery-public-data.google_analytics_sample.ga_sessions_*`

        WHERE _TABLE_SUFFIX BETWEEN '20170701' AND '20170731'

        AND totals.transactions IS not null

        GROUP BY fullVisitorId)

        SELECT SUM(num_transactions)/COUNT(*) AS avg_transactions_per_user

        FROM s

        """

df=bq_assistant.query_to_pandas_safe(query) 

df
# Q6. What is the average amount of money spent per session in July 2017?

# Session: The period of time a user is active on your site or app. 

# By default, if a user is inactive for 30 minutes or more, any future activity is attributed to a new session. 

# Users that leave your site and return within 30 minutes are counted as part of the original session.



query = """

        WITH s AS (SELECT totals.transactionRevenue AS total_spent

        FROM `bigquery-public-data.google_analytics_sample.ga_sessions_*`

        WHERE _TABLE_SUFFIX BETWEEN '20170701' AND '20170731'

        AND totals.transactionRevenue IS NOT NULL

        AND totals.transactionRevenue > 1)

        SELECT COUNT(*) AS Num_Session,SUM(total_spent)/COUNT(*) AS average_per_session

        FROM s

        """

bq_assistant.query_to_pandas_safe(query)
# Q7: What is the sequence of pages viewed?

query = """

        SELECT date, SUM(totals.pageviews) AS pageviews_per_day

        FROM `bigquery-public-data.google_analytics_sample.ga_sessions_*`

        WHERE _TABLE_SUFFIX BETWEEN '20170701' AND '20170731'

        GROUP BY date

        ORDER BY date

        """

df = bq_assistant.query_to_pandas_safe(query)
# Show the time sequence of pages viewed in 2017~2018 (simple pandas plot)

df.plot()