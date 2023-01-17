print('Importing 1 table for analysis')

import pandas as pd

from bq_helper import BigQueryHelper

bq_assistant = BigQueryHelper("bigquery-public-data", "google_analytics_sample")

bq_assistant.list_tables()[:1]
QUERY = """

    SELECT 

        *  -- Warning, be careful when doing SELECT ALL

    FROM `bigquery-public-data.google_analytics_sample.ga_sessions_20160801` 

    LIMIT 10

"""



df = bq_assistant.query_to_pandas(QUERY)

df.head()
schema = bq_assistant.table_schema("ga_sessions_20160801")

schema
schema[schema['name'].str.contains("totals")]
QUERY = """

    SELECT

        SUM(totals.visits) as visits

    FROM `bigquery-public-data.google_analytics_sample.ga_sessions_20160801` 

    LIMIT 10

"""



df = bq_assistant.query_to_pandas(QUERY)

df.head() 
QUERY = """

    SELECT

        SUM(totals.newVisits) as newVisits

    FROM `bigquery-public-data.google_analytics_sample.ga_sessions_20160801` 

    LIMIT 10

"""



df = bq_assistant.query_to_pandas(QUERY)

df.head() 
QUERY = """

    SELECT

        SUM(totals.uniqueScreenviews) as uscrview

    FROM `bigquery-public-data.google_analytics_sample.ga_sessions_20160801` 

    LIMIT 10

"""



df = bq_assistant.query_to_pandas(QUERY)

df.head() 
QUERY = """

    SELECT

        SUM(totals.visits) as visits

    FROM `bigquery-public-data.google_analytics_sample.ga_sessions_*` 

    WHERE _TABLE_SUFFIX BETWEEN '20170101'AND '20170131'

    LIMIT 10

"""



df = bq_assistant.query_to_pandas(QUERY)

df.head()
QUERY = """

    SELECT

        date as date,

        SUM(totals.visits) as visits

    FROM `bigquery-public-data.google_analytics_sample.ga_sessions_*` 

    WHERE _TABLE_SUFFIX BETWEEN '20170101'AND '20170131' 

    GROUP BY date

    ORDER BY date ASC

    LIMIT 31

"""



df = bq_assistant.query_to_pandas(QUERY)

df.head(31)
QUERY = """

    SELECT

        fullVisitorId

    FROM `bigquery-public-data.google_analytics_sample.ga_sessions_*` 

    WHERE _TABLE_SUFFIX BETWEEN '20170101'AND '20170131' 

    LIMIT 10

"""



df = bq_assistant.query_to_pandas(QUERY)

df.head()
# This way doesnt work!

QUERY = """

    SELECT

        fullVisitorId,

        COUNT(*) as the_count

    FROM `bigquery-public-data.google_analytics_sample.ga_sessions_*` 

    WHERE _TABLE_SUFFIX BETWEEN '20170101'AND '20170131' 

    GROUP BY fullVisitorId

    LIMIT 10

"""



df = bq_assistant.query_to_pandas(QUERY)

df.head(5)
# This way doesnt work!

QUERY = """

    SELECT

        COUNT(fullVisitorId) as the_count

    FROM `bigquery-public-data.google_analytics_sample.ga_sessions_*` 

    WHERE _TABLE_SUFFIX BETWEEN '20170101'AND '20170131' 

    LIMIT 10

"""



df = bq_assistant.query_to_pandas(QUERY)

df.head(5)
# This way works!

QUERY = """

    SELECT

        COUNT(DISTINCT fullVisitorId) as the_count

    FROM `bigquery-public-data.google_analytics_sample.ga_sessions_*` 

    WHERE _TABLE_SUFFIX BETWEEN '20170101'AND '20170131' 

    LIMIT 10

"""



df = bq_assistant.query_to_pandas(QUERY)

df.head(5)
schema[schema['name'].str.contains("page")]
QUERY = """

    SELECT

        date as date,

        SUM(totals.pageviews) as pageviews

    FROM `bigquery-public-data.google_analytics_sample.ga_sessions_*` 

    WHERE _TABLE_SUFFIX BETWEEN '20170101'AND '20170131' 

    GROUP BY date

    ORDER BY date ASC

    LIMIT 31

"""



df = bq_assistant.query_to_pandas(QUERY)

df.head()
schema[schema['name'].str.contains("Revenue")]
QUERY = """

    SELECT

        date as date,

        SUM(totals.transactionRevenue) as transactionRevenue

    FROM `bigquery-public-data.google_analytics_sample.ga_sessions_*` 

    WHERE _TABLE_SUFFIX BETWEEN '20170101'AND '20170131' 

    GROUP BY date

    ORDER BY date ASC

    LIMIT 31

"""



df = bq_assistant.query_to_pandas(QUERY)

df.head()
QUERY = """

    SELECT

        date as date,

        SUM(totals.transactionRevenue)/1e6 as transactionRevenue       ## 1e6 - means 10 to the power of 6 (or 1,000,000) AKA 10^6

    FROM `bigquery-public-data.google_analytics_sample.ga_sessions_*` 

    WHERE _TABLE_SUFFIX BETWEEN '20170101'AND '20170131' 

    GROUP BY date

    ORDER BY date ASC

    LIMIT 31

"""



df = bq_assistant.query_to_pandas(QUERY)

df.head(31)
schema[schema['name'].str.contains("custom")]
QUERY = """

    SELECT

        fullVisitorId,

        customDimensions

    FROM `bigquery-public-data.google_analytics_sample.ga_sessions_*` 

    WHERE _TABLE_SUFFIX BETWEEN '20170101'AND '20170131' 

    LIMIT 10

"""



df = bq_assistant.query_to_pandas(QUERY)

df.head(10)
schema[schema['name'].str.contains("custom")]

QUERY = """

    SELECT

        fullVisitorid,

        customDimensions,

        cds.Value,

        cds.Index

    FROM `bigquery-public-data.google_analytics_sample.ga_sessions_*`, 

        UNNEST(customDimensions) AS cds                                        #### this bit breaks out the bracket stuff

    WHERE _TABLE_SUFFIX BETWEEN '20170101'AND '20170131' 

    AND index = 4

    LIMIT 10

"""



df = bq_assistant.query_to_pandas(QUERY)

df.head()
QUERY = """

    SELECT

        SUM(totals.transactionRevenue) as trans_revenue

    FROM `bigquery-public-data.google_analytics_sample.ga_sessions_*` 

    WHERE _TABLE_SUFFIX BETWEEN '20170601'AND '20170630'

    LIMIT 10

"""



df = bq_assistant.query_to_pandas(QUERY)

df.head()
schema[schema['name'].str.contains("traffic")]
QUERY = """

    SELECT

        totals.transactionRevenue,

        trafficSource,

        ts.referralPath,

        ts.campaign,

        ts.source

    FROM `bigquery-public-data.google_analytics_sample.ga_sessions_*`, 

        UNNEST(trafficSource) AS ts

    WHERE _TABLE_SUFFIX BETWEEN '20170601'AND '20170630' 

    ORDER BY totals.transactionRevenue ASC

    LIMIT 10

"""



df = bq_assistant.query_to_pandas(QUERY)

df.head()
QUERY = """

  SELECT 

      fullVisitorId, 

      TIMESTAMP_SECONDS(visitStartTime) AS visitStartTime,

      TIMESTAMP_ADD(TIMESTAMP_SECONDS(visitStartTime), INTERVAL hits.time MILLISECOND) AS hitTime,

      hits.page.pagePath,

      hits.type

  FROM 

      `bigquery-public-data.google_analytics_sample.ga_sessions_*` AS ga, 

      unnest(hits) as hits

  WHERE fullVisitorId = '0509972280802528263' AND

      _TABLE_SUFFIX BETWEEN '20170801' AND '20170801'

  ORDER BY hitTime 

  LIMIT 50

  """

df = bq_assistant.query_to_pandas(QUERY)

df
QUERY = """  

  SELECT 

      fullVisitorId, 

      TIMESTAMP_SECONDS(visitStartTime) AS visitStartTime,

      TIMESTAMP_ADD(TIMESTAMP_SECONDS(visitStartTime), INTERVAL hits.time MILLISECOND) AS hitTime,

      hits.page.pagePath,

      hits.type

  FROM 

      `bigquery-public-data.google_analytics_sample.ga_sessions_*` AS ga, 

      unnest(hits) as hits

  WHERE _TABLE_SUFFIX BETWEEN '20170801' AND '20170801'

  ORDER BY fullVisitorId, hitTime 

  LIMIT 500

  """

df = bq_assistant.query_to_pandas(QUERY)

df