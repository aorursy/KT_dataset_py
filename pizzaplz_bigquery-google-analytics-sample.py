import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline  


from google.cloud import bigquery
import pandas as pd
# initiate bigquery client
int_query = bigquery.Client()
query1 = """
SELECT *
FROM `bigquery-public-data.google_analytics_sample.ga_sessions_*` ga
WHERE _TABLE_SUFFIX BETWEEN '20170701' AND '20170731'
limit 10
        """

data = int_query.query(query1).to_dataframe()
data.head()
query1 = """
SELECT
geonetwork
FROM `bigquery-public-data.google_analytics_sample.ga_sessions_*`
WHERE _TABLE_SUFFIX BETWEEN '20170701' AND '20170731'
limit 10

        """

data = int_query.query(query1).to_dataframe()

#Will expand width of the column to see entire value of the field
pd.set_option('display.max_colwidth',-1)
data.head()
query1 = """
SELECT
distinct geonetwork.country
FROM `bigquery-public-data.google_analytics_sample.ga_sessions_*`
WHERE _TABLE_SUFFIX BETWEEN '20170701' AND '20170731'
order by 1 desc

        """

data = int_query.query(query1).to_dataframe()
pd.set_option('display.max_colwidth',-1)
data.head(15)
query1 = """
SELECT
totals
FROM `bigquery-public-data.google_analytics_sample.ga_sessions_*`
WHERE _TABLE_SUFFIX BETWEEN '20170701' AND '20170731'
limit 10

        """

data = int_query.query(query1).to_dataframe()

#Will expand width of the column to see entire value of the field
pd.set_option('display.max_colwidth',-1)
data.head(10)
query1 = """
SELECT
trafficSource
FROM `bigquery-public-data.google_analytics_sample.ga_sessions_*`
WHERE _TABLE_SUFFIX BETWEEN '20170701' AND '20170731'
limit 10

        """

data = int_query.query(query1).to_dataframe()

#Will expand width of the column to see entire value of the field
pd.set_option('display.max_colwidth',-1)
data.head(3)
#Resetting default options

#resets the display options which were modified earlier to view JSON fields
pd.reset_option('^display.')

#resets every option
#pd.reset_option('all')
transactions = """
SELECT
device.browser,
geonetwork.country,
sum(totals.transactions) as total_transactions_value
FROM `bigquery-public-data.google_analytics_sample.ga_sessions_*`
WHERE _TABLE_SUFFIX BETWEEN '20170701' AND '20170731'
AND geonetwork.country = 'United States'
group by 1,2
order by 3 desc

        """

data = int_query.query(transactions).to_dataframe()
data.head(10)

bounce_rate = """
SELECT
trafficSource.source,
sum(case when totals.pageviews = 1 then totals.visits else 0 end) as bounce_visits,
sum(totals.visits) as total_visits,
sum(case when totals.pageviews = 1 then totals.visits else 0 end) / sum(totals.visits) as bounce_rate
FROM `bigquery-public-data.google_analytics_sample.ga_sessions_*`
WHERE _TABLE_SUFFIX BETWEEN '20170701' AND '20170731'
AND geonetwork.country = 'United States'
group by 1
order by 3 desc

        """

data = int_query.query(bounce_rate).to_dataframe()
data.head()
pageview_transactions = """
SELECT
distinct totals.transactions,
avg(totals.pageviews) as avg_pageviews
FROM `bigquery-public-data.google_analytics_sample.ga_sessions_*`
WHERE _TABLE_SUFFIX BETWEEN '20170701' AND '20170731'
AND geonetwork.country = 'United States'
AND totals.transactionRevenue is not null
group by 1
order by 1 asc


        """

data = int_query.query(pageview_transactions).to_dataframe()
data.head()
pageview_notransactions = """
SELECT
distinct totals.transactions,
avg(totals.pageviews) as avg_pageviews
FROM `bigquery-public-data.google_analytics_sample.ga_sessions_*`
WHERE _TABLE_SUFFIX BETWEEN '20170701' AND '20170731'
AND geonetwork.country = 'United States'
AND totals.transactionRevenue is null
group by 1
order by 1 asc


        """

data = int_query.query(pageview_notransactions).to_dataframe()
data.head()
avg_transaction = """
SELECT
avg(totals.transactions) as avg_transaction
FROM `bigquery-public-data.google_analytics_sample.ga_sessions_*`
WHERE _TABLE_SUFFIX BETWEEN '20170701' AND '20170731'
AND geonetwork.country = 'United States'
AND totals.transactionRevenue is not null
limit 100
        """

data = int_query.query(avg_transaction).to_dataframe()
data

avg_transaction = """
SELECT
distinct visitNumber,
avg(totals.totalTransactionRevenue) as avg_transaction
FROM `bigquery-public-data.google_analytics_sample.ga_sessions_*`
WHERE _TABLE_SUFFIX BETWEEN '20170701' AND '20170731'
AND geonetwork.country = 'United States'
#AND totals.totalTransactionRevenue is not null
group by 1
order by 2 desc
limit 100
        """

data = int_query.query(avg_transaction).to_dataframe()
data.head()