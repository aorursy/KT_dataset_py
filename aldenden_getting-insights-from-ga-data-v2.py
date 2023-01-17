print('What tables do I have?')
import pandas as pd
from bq_helper import BigQueryHelper
bq_assistant = BigQueryHelper("bigquery-public-data", "google_analytics_sample")
bq_assistant.list_tables()[:5]
QUERY = """
    SELECT 
        *  -- Warning, be careful when doing SELECT ALL
    FROM `bigquery-public-data.google_analytics_sample.ga_sessions_20160801` 
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
QUERY = """
  WITH user_hit AS
  (SELECT 
      fullVisitorId, 
      TIMESTAMP_SECONDS(visitStartTime) AS visitStartTime,
      TIMESTAMP_ADD(TIMESTAMP_SECONDS(visitStartTime), INTERVAL hits.time MILLISECOND) AS hitTime,
      hits.page.pagePath,
      REGEXP_EXTRACT(hits.page.pagePath,r"(?:\/[^/?&#]*){0,1}") AS pathCategory
  FROM 
      `bigquery-public-data.google_analytics_sample.ga_sessions_*` AS ga, 
      unnest(hits) as hits
  WHERE _TABLE_SUFFIX BETWEEN '20170801' AND '20170801' 
      AND hits.type="PAGE"
  ORDER BY hitTime)
  
  SELECT pathCategory, COUNT(*) AS count 
  FROM user_hit
  GROUP BY 1;
  """
df = bq_assistant.query_to_pandas(QUERY)
df
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import pyplot
import seaborn as sns

plt.figure();
df = df.set_index('pathCategory')
df['count'].plot(kind='bar');
query = """
    SELECT
        fullVisitorId,
        EXTRACT(DATE FROM TIMESTAMP_SECONDS(visitStartTime)) AS date,
        SUM(totals.totalTransactionRevenue)/1e6 AS revenue
    FROM
        `bigquery-public-data.google_analytics_sample.ga_sessions_*`
    WHERE
        _TABLE_SUFFIX BETWEEN '20170701' AND '20170731'
    GROUP BY
        fullVisitorId,
        date
    HAVING revenue > 0
"""
df = bq_assistant.query_to_pandas(query)
df.head(10)
#https://lifetimes.readthedocs.io/en/latest/index.html
!pip install lifetimes
from lifetimes.utils import summary_data_from_transaction_data
from lifetimes import BetaGeoFitter

effective_date = "2017-07-31"
summary = summary_data_from_transaction_data(df, 'fullVisitorId', 'date', 
                                             observation_period_end=effective_date)
summary['evaluate_date'] = effective_date
bgf = BetaGeoFitter(penalizer_coef=0.0) ## this is one of the models we can use :)
bgf.fit(summary['frequency'], summary['recency'], summary['T'])
t = 7 #predict purchases in 7 days
summary['predicted_purchases'] = bgf.predict(t, summary['frequency'], summary['recency'], summary['T'])

from lifetimes.plotting import plot_period_transactions
plot_period_transactions(bgf)
summary[summary['frequency']>1].head()
from lifetimes.plotting import plot_history_alive

id = '2158257269735455737'
days_since_birth = 60
sp_trans = df.loc[df['fullVisitorId'] == id]
plot_history_alive(bgf, days_since_birth, sp_trans, 'date')