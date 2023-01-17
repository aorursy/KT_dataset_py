print('What tables do I have?')

import pandas as pd

from bq_helper import BigQueryHelper

bq_assistant = BigQueryHelper("bigquery-public-data", "google_analytics_sample")

bq_assistant.list_tables()
QUERY = """

    SELECT 

        *  -- Warning, be careful when doing SELECT ALL

    FROM `bigquery-public-data.google_analytics_sample.ga_sessions_20160801` 

    LIMIT 10

"""



df = bq_assistant.query_to_pandas(QUERY)

df.head()
bq_assistant.table_schema("ga_sessions_20170128")
QUERY = """

    SELECT

    SUM (totals.visits) AS totals_visits

    FROM `bigquery-public-data.google_analytics_sample.ga_sessions_20170128` 



"""



df = bq_assistant.query_to_pandas(QUERY)

df.head()
QUERY = """

    SELECT 

    SUM (totals.visits) AS totals_visits_for_jan17

    FROM `bigquery-public-data.google_analytics_sample.ga_sessions_*`

    WHERE date BETWEEN '20170101' AND '20170131'

     



"""



df = bq_assistant.query_to_pandas(QUERY)

df.head()
QUERY = """

    SELECT

    date,

    SUM (totals.visits) AS totals_visits_for_jan17_perDay

    FROM `bigquery-public-data.google_analytics_sample.ga_sessions_*`

    WHERE date BETWEEN '20170101' AND '20170131'

    GROUP BY date

    ORDER BY date ASC

    

"""



df = bq_assistant.query_to_pandas(QUERY)

df.head(50)

QUERY = """

    SELECT

    date,

    SUM(totals.totalTransactionRevenue)/1000000 AS Revenue

    FROM `bigquery-public-data.google_analytics_sample.ga_sessions_*`

    WHERE date BETWEEN '20170101' AND '20170131'

    GROUP BY date

    ORDER BY SUM(totals.totalTransactionRevenue) DESC



    

"""



df = bq_assistant.query_to_pandas(QUERY)

df.head(50)
QUERY = """



SELECT

SUM(totals.totalTransactionRevenue)/1000000 AS revenue_in_Jun17

FROM `bigquery-public-data.google_analytics_sample.ga_sessions_*`

WHERE date BETWEEN '20170601' AND '20170630'

 

    

"""



df = bq_assistant.query_to_pandas(QUERY)

df.head()
QUERY = """



SELECT

 trafficSource.source,

 SUM(totals.totalTransactionRevenue)/1000000 AS Revenue

 

    FROM `bigquery-public-data.google_analytics_sample.ga_sessions_*`

    WHERE date BETWEEN '20170601' AND '20170630'

    

     GROUP BY

     trafficSource.source

     ORDER BY

     Revenue DESC

LIMIT 10;

"""

df = bq_assistant.query_to_pandas(QUERY)

df.head(10)



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

  WHERE date BETWEEN '20170801' AND '20170801'

  ORDER BY fullVisitorId, hitTime 

  LIMIT 500

  """

df = bq_assistant.query_to_pandas(QUERY)

df