print('What data do I have?')
import pandas as pd
from bq_helper import BigQueryHelper
bq_assistant = BigQueryHelper("bigquery-public-data", "google_analytics_sample")
bq_assistant.list_tables()
QUERY = "SELECT * FROM `bigquery-public-data.google_analytics_sample.ga_sessions_20160801` LIMIT 10"
df = bq_assistant.query_to_pandas(QUERY)
df.head()
QUERY = """
  SELECT 
      fullVisitorId, 
      visitStartTime, 
      ANY_VALUE(CONCAT(trafficSource.source,'/',trafficSource.medium)) AS sourceMedium,
      SUM(CAST(hits.eCommerceAction.action_type AS INT64)) AS activeInteractions,
      ANY_VALUE(totals.totalTransactionRevenue/1e6) AS txnRevenue
  FROM 
      `bigquery-public-data.google_analytics_sample.ga_sessions_20170801` AS ga, 
      unnest(hits) as hits
  GROUP BY 
      fullVisitorId, visitStartTime 
  HAVING txnRevenue IS NOT NULL
  LIMIT 10
  """
df = bq_assistant.query_to_pandas(QUERY)
df.head()
query1 = """SELECT
device.browser,
SUM ( totals.transactions ) AS total_transactions
FROM `bigquery-public-data.google_analytics_sample.ga_sessions_*`
WHERE
_TABLE_SUFFIX BETWEEN '20170701' AND '20170731'
GROUP BY
device.browser
ORDER BY
total_transactions DESC;
        """
response1 = bq_assistant.query_to_pandas(query1)
response1.head(10)
query2 = """SELECT
source,
total_visits,
total_no_of_bounces,
( ( total_no_of_bounces / total_visits ) * 100 ) AS bounce_rate
FROM (
SELECT
trafficSource.source AS source,
COUNT ( trafficSource.source ) AS total_visits,
SUM ( totals.bounces ) AS total_no_of_bounces
FROM `bigquery-public-data.google_analytics_sample.ga_sessions_*`
WHERE
_TABLE_SUFFIX BETWEEN '20170701' AND '20170731'
GROUP BY
source )
ORDER BY
total_visits DESC
LIMIT 10;
        """
response2 = bq_assistant.query_to_pandas(query2)
response2.head(10)
query6 = """SELECT
( SUM(total_transactionrevenue_per_user) / SUM(total_visits_per_user) ) AS
avg_revenue_by_user_per_visit
FROM (
SELECT
fullVisitorId,
SUM( totals.visits ) AS total_visits_per_user,
SUM( totals.transactionRevenue ) AS total_transactionrevenue_per_user
FROM
`bigquery-public-data.google_analytics_sample.ga_sessions_*`
WHERE
_TABLE_SUFFIX BETWEEN '20170701' AND '20170731'
AND
totals.visits > 0
AND totals.transactions >= 1
AND totals.transactionRevenue IS NOT NULL
GROUP BY
fullVisitorId
LIMIT 10);
        """
response6 = bq_assistant.query_to_pandas(query6)
response6.head(10)
query7 = """SELECT
fullVisitorId,
visitId,
visitNumber,
hits.hitNumber AS hitNumber,
hits.page.pagePath AS pagePath
FROM
`bigquery-public-data.google_analytics_sample.ga_sessions_*`,
UNNEST(hits) as hits
WHERE
_TABLE_SUFFIX BETWEEN '20170701' AND '20170731'
AND
hits.type="PAGE"
ORDER BY
fullVisitorId,
visitId,
visitNumber,
hitNumber
LIMIT 10;
        """
response7 = bq_assistant.query_to_pandas(query7)
response7.head(10)