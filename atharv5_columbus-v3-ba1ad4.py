print('What tables do I have?')

import pandas as pd

from bq_helper import BigQueryHelper

bq_assistant = BigQueryHelper("bigquery-public-data", "google_analytics_sample")

bq_assistant.list_tables()[:5]
QUERY = """

    SELECT 

        *  -- Warning, be careful when doing SELECT ALL

    FROM `bigquery-public-data.google_analytics_sample.ga_sessions_20160801` 

    LIMIT 100

"""



df = bq_assistant.query_to_pandas(QUERY)

df.head()
import pandas as pd

from bq_helper import BigQueryHelper

bq_assistant = BigQueryHelper("bigquery-public-data", "google_analytics_sample")

schema = bq_assistant.table_schema('ga_sessions_20170128')

QUERY = "SELECT SUM(totals.visits) as visits FROM `bigquery-public-data.google_analytics_sample.ga_sessions_20170128`"

df = bq_assistant.query_to_pandas(QUERY)

df.head()

import pandas as pd

from bq_helper import BigQueryHelper

bq_assistant = BigQueryHelper("bigquery-public-data", "google_analytics_sample")

QUERY = "SELECT SUM (totals.visits) as Visits FROM `bigquery-public-data.google_analytics_sample.ga_sessions_*` WHERE _TABLE_SUFFIX BETWEEN '20170101' AND '20170131'"

df = bq_assistant.query_to_pandas(QUERY)

df.head()

import pandas as pd

from bq_helper import BigQueryHelper

bq_assistant = BigQueryHelper("bigquery-public-data", "google_analytics_sample")

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

import pandas as pd

from bq_helper import BigQueryHelper

bq_assistant = BigQueryHelper("bigquery-public-data", "google_analytics_sample")

QUERY = """

    SELECT

        date as date,

        SUM(totals.transactionRevenue) as transactionRevenue

    FROM `bigquery-public-data.google_analytics_sample.ga_sessions_*` 

    WHERE _TABLE_SUFFIX BETWEEN '20170101'AND '20170131' 

    GROUP BY date

    ORDER BY transactionRevenue DESC

    LIMIT 31

"""

df = bq_assistant.query_to_pandas(QUERY)

df.head(31)

import pandas as pd

from bq_helper import BigQueryHelper

bq_assistant = BigQueryHelper("bigquery-public-data", "google_analytics_sample")

QUERY = """

    SELECT

        

        SUM(totals.transactionRevenue) as totalRevenue

    FROM `bigquery-public-data.google_analytics_sample.ga_sessions_*` 

    WHERE _TABLE_SUFFIX BETWEEN '20170601'AND '20170630' 

"""

df = bq_assistant.query_to_pandas(QUERY)

df.head(31)

import pandas as pd

from bq_helper import BigQueryHelper

bq_assistant = BigQueryHelper("bigquery-public-data", "google_analytics_sample")

schema = bq_assistant.table_schema('ga_sessions_20170630')

QUERY = """

    SELECT

        trafficSource.source as Source ,totals.transactionRevenue as Revenue

    FROM `bigquery-public-data.google_analytics_sample.ga_sessions_*` 

    WHERE _TABLE_SUFFIX BETWEEN '20170601'AND '20170630' 

"""

df = bq_assistant.query_to_pandas(QUERY)

df.head() 
import pandas as pd

from bq_helper import BigQueryHelper

bq_assistant = BigQueryHelper("bigquery-public-data", "google_analytics_sample")

schema = bq_assistant.table_schema('ga_sessions_20170630')

QUERY = """

  SELECT 

      fullVisitorId, 

      TIMESTAMP_SECONDS(visitStartTime) AS visitStartTime,

      TIMESTAMP_ADD(TIMESTAMP_SECONDS(visitStartTime), INTERVAL hits.time MILLISECOND) AS hitTime,

      hits.page.pagePath,

      hits.type

  FROM 

      `bigquery-public-data.google_analytics_sample.ga_sessions_20170108` AS ga, 

      unnest(hits) as hits

  ORDER BY hitTime 

  LIMIT 50

  """



df = bq_assistant.query_to_pandas(QUERY)

df.head() 