import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from  bq_helper import BigQueryHelper as bqh
bqa = bqh('bigquery-public-data','google_analytics_sample')
bqa.table_schema("ga_sessions_20160801")
QUERY = """
    SELECT
        date as Date,
        SUM(totals.transactionRevenue)/1000000 as Revenue
    FROM 
      `bigquery-public-data.google_analytics_sample.ga_sessions_*` AS ga
    WHERE _TABLE_SUFFIX BETWEEN '20160801' AND '20170801'
    GROUP BY Date
    ORDER BY Date ASC
"""

data = bqa.query_to_pandas(QUERY)
data.Revenue=data.Revenue.fillna(0)
data.Date = data.Date.apply(lambda x: pd.to_datetime(str(x), format='%Y%m%d'))

print('Days of data: ',len(data))
print('Total Revenue: $',end='')
print("{:,}".format(data.Revenue.sum()))
data
QUERY = """
    SELECT
        trafficSource.source as Source,
        SUM(totals.transactionRevenue)/1000000 as Revenue,
        AVG(totals.timeOnSite) as Time,
        AVG(totals.transactionRevenue)/1000000 as AverageRevenue
    FROM `bigquery-public-data.google_analytics_sample.ga_sessions_*` 
    WHERE
        _TABLE_SUFFIX BETWEEN '20160801'AND '20170801' AND totals.transactions > 0
    GROUP BY Source
    ORDER BY Revenue DESC
"""

datarevenue = bqa.query_to_pandas(QUERY)

datarevenue.drop(columns=['Time']).head(10)
QUERY =  """
    SELECT
        fullVisitorId,
        trafficSource.source as TrafficSource,
        device.browser as Browser,
        device.deviceCategory as Device,
        geoNetwork.country as Country,
        SUM(totals.transactionRevenue)/1000000 as Revenue
    FROM 
      `bigquery-public-data.google_analytics_sample.ga_sessions_*` AS ga
    WHERE _TABLE_SUFFIX BETWEEN '20160801' AND '20170801'
    GROUP BY fullVisitorId,TrafficSource,Browser,Device,Country
    ORDER BY Revenue DESC
    LIMIT 10
"""

data = bqa.query_to_pandas(QUERY)
data
!pip install psycopg2
!pip install pandas_gbq
from pandas.io import gbq
from google.oauth2 import service_account
from google.cloud import bigquery
from google.cloud.bigquery import Dataset
import psycopg2
import pandas_gbq as bq
import timeit
import numpy as np
import seaborn as sns

project_id = "portofolio-285302"
query = """
SELECT channelGrouping channel, device.browser browser, geoNetwork.country country, prod.productListPosition position, hits.page.pagePath path, prod.v2ProductCategory category,  trafficSource.source as TrafficSource, prod.productPrice price, prod.productListName listname, hits.promotionActionInfo.promoIsView, hits.promotionActionInfo.promoIsClick,
SUM(totals.pageviews) pageviews,  SUM(totals.timeOnSite) timeOnSite, SUM(totals.totalTransactionRevenue)/1000000 revenue, SUM(totals.transactions) transaction, COUNT(totals.hits) hits

FROM `bigquery-public-data.google_analytics_sample.ga_sessions_*`, UNNEST(hits) as hits, UNNEST(hits.product) as prod
GROUP BY 1,2,3,4,5,6,7,8,9,10,11
"""

dialect = "standard"
df = pd.read_gbq(query,project_id, dialect = dialect)

corr_matrix = df.corr()
mask = np.zeros_like(corr_matrix, dtype=np.bool)
mask[np.triu_indices_from(mask)]= True
import seaborn as sns
f, ax = plt.subplots(figsize=(22, 30))
heatmap = sns.heatmap(corr_matrix,
square = True,
mask = mask,
linewidths = .5,
cmap = 'coolwarm',
cbar_kws = {'shrink': .4, 'ticks' : [-1, -.5, 0, 0.5, 1]},
vmin = -1,
vmax = 1,
annot = True,
annot_kws = {'size': 8})
ax.set_yticklabels(corr_matrix.columns, rotation = 0)
ax.set_xticklabels(corr_matrix.columns)
sns.set_style({'xtick.bottom': True}, {'ytick.left': True})