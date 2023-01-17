

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import bq_helper



# https://www.kaggle.com/sohier/introduction-to-the-bq-helper-package

bq_assistant = bq_helper.BigQueryHelper("bigquery-public-data", "google_analytics_sample")

print(bq_assistant.list_tables()[:5])

print(bq_assistant.list_tables()[-5:])

table_names = bq_assistant.list_tables()

# a table for each day : 2016-08-01 - 2017-08-01

# a year's worth of data
bq_assistant.head("ga_sessions_20160801", num_rows=5)

# Many nested columns 

    # BiQuery allows access via "category_name.subcategory_name"

# What do rows represent?

    # Each row within a table corresponds to a session in Analytics 360.

last_schema = bq_assistant.table_schema("ga_sessions_20170801")

first_col_names = bq_assistant.table_schema("ga_sessions_20160801")['name']

last_col_names = bq_assistant.table_schema("ga_sessions_20170801")['name']

print("Number of columns in 2016:", len(first_col_names))

print("Number of columns in 2017:", len(last_col_names))
# print new columns 

[c for c in last_col_names if c not in first_col_names.tolist()]
def inspect(query, nrows=15, sample=False):

    """Display response from given query but don't save. 

    query: str, raw SQL query

    nrows: int, number of rows to display, default 15

    sample: bool, use df.sample instead of df.head, default False """

    response = bq_assistant.query_to_pandas_safe(query, max_gb_scanned=10)

    if sample:

        return response.sample(nrows)

    return response.head(nrows) 



def retrieve(query, nrows=10):

    """Save response from given query and print a preview. 

    query: str, raw SQL query

    nrows: int, number of rows to display"""

    response = bq_assistant.query_to_pandas_safe(query, max_gb_scanned=10)

    print(response.head(nrows))

    return response
query = """

SELECT

    COUNT(*)

FROM

    `bigquery-public-data.google_analytics_sample.ga_sessions_20160801`

"""

inspect(query)
query = """

SELECT COUNT(*)

FROM 

     (SELECT DISTINCT fullVisitorId, visitID

        FROM

    `bigquery-public-data.google_analytics_sample.ga_sessions_20160801`) s

"""

inspect(query)
# Total Sessions

query = """

SELECT

    COUNT(*)

FROM

    `bigquery-public-data.google_analytics_sample.ga_sessions_*`

"""

inspect(query)

# The wild card parses to a UNION ALL
# Unique sessions 

query = """

SELECT COUNT(*)

FROM 

     (SELECT DISTINCT fullVisitorId, visitID

        FROM

    `bigquery-public-data.google_analytics_sample.ga_sessions_*`) s

"""

inspect(query)



# A few(~900) sessions don't have unique ID's if looking across all tables
# Quick Check of alternative method Unique sessions 

query = """

SELECT COUNT(*)

FROM 

     (SELECT DISTINCT concat(fullVisitorId, cast(visitID as string))

        FROM

    `bigquery-public-data.google_analytics_sample.ga_sessions_*`) s

"""

inspect(query)
# Test alt way to count distinct 

table = table_names[0]

print(table)

date = table[-8:]

print(date)

query = f"""

SELECT {date} as table, 

COUNT(*) as total, 

count(DISTINCT CONCAT(fullVisitorId, CAST(visitID as string))) as unique

FROM 

    `bigquery-public-data.google_analytics_sample.{table}`

"""

inspect(query)
# ' UNION '.join(['A', 'B', 'C'])
# Create a bunch of queries with a loop

# queries = []

# for table in table_names:

#     date = table[-8:]

#     query = f"""SELECT {date}, COUNT(*) as total, count(DISTINCT CONCAT(fullVisitorId, CAST(visitID as string))) as unique FROM `bigquery-public-data.google_analytics_sample.{table}`"""

#     queries.append(query)

# queries[:5]
# %%time # wall time 9 mins



# tables = []

# mismatches = []

# for query in queries:

#     table_info = bq_assistant.query_to_pandas_safe(query)

#     tables.append(table_info)

#     if table_info.loc[0, 'total'] != table_info.loc[0, 'unique']:

#         print(table_info['date'])

#         mismatches.append(table_info)
# # List of tables with record counts

# table_records = pd.concat(tables, axis=0)

# table_records
# Test unnesting- needed since it's a list within the column

unnested_query = """

SELECT

    fullVisitorId,

    visitId,

    visitNumber,

    hits.hitNumber AS hitNumber,

    hits.page.pagePath AS pagePath

FROM

    `bigquery-public-data.google_analytics_sample.ga_sessions_20160801`, UNNEST(hits) as hits

WHERE

    hits.type="PAGE"

ORDER BY

    fullVisitorId,

    visitId,

    visitNumber,

    hitNumber

"""

unnested_query_df = retrieve(unnested_query)

unnested_query_df



# How many unique visitors ?

query = """ SELECT COUNT (DISTINCT fullVisitorId) unique_visitors

            FROM `bigquery-public-data.google_analytics_sample.ga_sessions_*`            

            """

inspect(query)
# How many unique customers ?

query = """ SELECT COUNT (DISTINCT fullVisitorId) unique_visitors

            FROM `bigquery-public-data.google_analytics_sample.ga_sessions_*`

            WHERE totals.transactions > 0 

            """

inspect(query)
# Repeat customers and how many sessions with a purchase

query = """ SELECT fullVisitorId, COUNT(DISTINCT visitId) cnt_purchases

            FROM `bigquery-public-data.google_analytics_sample.ga_sessions_*`

            WHERE totals.transactions > 0 

            GROUP BY fullVisitorId

            HAVING COUNT(DISTINCT visitId) > 1

            """

        

repeat_customers = retrieve(query)

print(repeat_customers.shape)

repeat_customers['cnt_purchases'].sum()

# group by device browser from july tables 

query = f"""

SELECT device.browser, 

    COUNT(totals.totalTransactionRevenue) AS cnt_revenue,

    COUNT(totals.transactions) AS cnt_transactions, 

    COUNT(*) AS cnt_all_rows, --sessions with browser

    COUNT(totals.transactions) / COUNT(*) AS sess_conversion_rate,

    sum(totals.totalTransactionRevenue) AS sum_revenue, 

    Sum(totals.transactions) AS  sum_transactions,

    sum(totals.totalTransactionRevenue) / Sum(totals.transactions) AS revenue_per_transaction

FROM `bigquery-public-data.google_analytics_sample.ga_sessions_201707*` 

 GROUP BY device.browser

 ORDER BY sum_transactions DESC

"""



transactions_by_browser = retrieve(query)
transactions_by_browser.head(10)

# Most transactions are done through Chrome. It also has the highest session-to-transaction conversion rate and revenue per transaction.
# "Real" bounce rate -- go by definition and filter where total.pageviews = 1 and divide by total visits

query = """

SELECT t.source,

    t.total_visits,

    b.bounce_visits,

    100 * b.bounce_visits / t.total_visits AS bounce_rate

FROM 

(SELECT trafficSource.source, COUNT(visitId) AS total_visits 

FROM `bigquery-public-data.google_analytics_sample.ga_sessions_201707*`

GROUP BY trafficSource.source) t



JOIN (SELECT trafficSource.source, COUNT(visitId) bounce_visits 

FROM `bigquery-public-data.google_analytics_sample.ga_sessions_201707*`

WHERE totals.pageviews = 1

GROUP BY trafficSource.source) b

ON t.source = b.source

ORDER by total_visits DESC



"""

inspect(query)



# Slightly different results than using totals.bounces column - more bounces visits but total visits are the same
# GA bounce rate 

query = """

SELECT trafficSource.source, 

    COUNT(visitId) AS total_visits,

    COUNT(totals.bounces) AS bounce_visits,

    100 * COUNT(totals.bounces) / COUNT(visitId) AS bounce_rate

FROM `bigquery-public-data.google_analytics_sample.ga_sessions_201707*`

GROUP BY trafficSource.source

ORDER BY total_visits DESC

"""



inspect(query)
query = """

SELECT DISTINCT totals.pageviews AS pageview_values

FROM `bigquery-public-data.google_analytics_sample.ga_sessions_201707*`

WHERE totals.bounces= 1

"""

inspect(query)
query = """

SELECT DISTINCT totals.bounces AS bounce_values

FROM `bigquery-public-data.google_analytics_sample.ga_sessions_201707*`

WHERE totals.pageviews= 1

"""

inspect(query)



# total.bounces column - for convenience but is sometimes null for a session with 1 pageview -- explains the extra bounces 
# Cofirm same results as other kernel

howto_query = """SELECT

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

total_visits DESC;

        """

inspect(howto_query)
# get users who made a purchase in July 2017

query = """

SELECT DISTINCT fullVisitorId 

FROM `bigquery-public-data.google_analytics_sample.ga_sessions_201707*`

WHERE totals.transactions > 0

"""

inspect(query)
# more info on users who made a purchase in July 2017

query = """

SELECT 

    COUNT(fullVisitorId) AS total_visitors, 

    COUNT(DISTINCT fullVisitorId) AS unique_visitors, 

    SUM(totals.transactions) AS sum_transactions,

    SUM(totals.totalTransactionRevenue) AS sum_revenue,

    AVG(totals.pageviews) as avg_pageviews -- in sessions where purchase was made



FROM `bigquery-public-data.google_analytics_sample.ga_sessions_201707*`

WHERE totals.transactions > 0

"""

inspect(query)
# calculate average among those users who made a purchase in July 2017

query = """

SELECT AVG(totals.pageviews) as avg_pageviews -- includes other sessions by user in which they did not make a purchase

 FROM `bigquery-public-data.google_analytics_sample.ga_sessions_201707*`

WHERE fullVisitorId IN (SELECT DISTINCT fullVisitorId 

                         FROM `bigquery-public-data.google_analytics_sample.ga_sessions_201707*`

                        WHERE totals.transactions > 0)

"""

inspect(query)
query = """

SELECT 

    COUNT(fullVisitorId) AS total_visitors, 

    COUNT(DISTINCT fullVisitorId) AS unique_visitors, 

    SUM(totals.transactions) AS sum_transactions,

    SUM(totals.totalTransactionRevenue) AS sum_revenue,

    AVG(totals.pageviews) as avg_pageviews -- in sessions where purchase were not made



FROM `bigquery-public-data.google_analytics_sample.ga_sessions_201707*`

WHERE totals.transactions IS NULL

"""

inspect(query)
# calculate average among those users who did not made a purchase in July 2017

query = """

SELECT AVG(totals.pageviews) as avg_pageviews -- includes other sessions by user in which they did make a purchase

 FROM `bigquery-public-data.google_analytics_sample.ga_sessions_201707*`

WHERE fullVisitorId IN (SELECT DISTINCT fullVisitorId 

                         FROM `bigquery-public-data.google_analytics_sample.ga_sessions_201707*`

                        WHERE totals.transactions IS NULL)

"""

inspect(query)
query = """

SELECT 

    COUNT(fullVisitorId) AS total_visitors, 

    COUNT(DISTINCT fullVisitorId) AS unique_visitors, 

    SUM(totals.transactions) AS sum_transactions,

    AVG(totals.totalTransactionRevenue) AS sum_revenue,

    AVG(totals.transactions) as avg_transactions, -- in sessions where purchase were  made

    SUM(totals.transactions) / COUNT(fullVisitorId) as alt_avg_transactions

FROM `bigquery-public-data.google_analytics_sample.ga_sessions_201707*`

WHERE totals.transactions > 0

"""

inspect(query)
# calculate average among those users who made a purchase in July 2017

query = """

SELECT AVG(totals.transactions) as avg_transactions,  -- includes other sessions by user in which they did not make a purchase

      SUM(totals.transactions) / COUNT(fullVisitorId) as alt_avg_transactions -- AVG function ignores NULLs by default

 FROM `bigquery-public-data.google_analytics_sample.ga_sessions_201707*`

WHERE fullVisitorId IN (SELECT DISTINCT fullVisitorId 

                         FROM `bigquery-public-data.google_analytics_sample.ga_sessions_201707*`

                        WHERE totals.transactions > 0)

"""

inspect(query)
query = """

SELECT 

    AVG(totals.totalTransactionRevenue) as avg_money_spent_per_purchase,

    SUM(totals.totalTransactionRevenue) / COUNT(*) as avg_money_spent

 FROM `bigquery-public-data.google_analytics_sample.ga_sessions_201707*`

"""

inspect(query)
 

howto_query6 = """SELECT

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

fullVisitorId );

"""

inspect(howto_query6)

# Why group by VisitorId (ie user) when it asks for money spent per session? 
query = """

SELECT 

   COUNT(totals.visits) as visits,

   COUNT(*) - COUNT(totals.visits) as sessions_not_visits

 FROM `bigquery-public-data.google_analytics_sample.ga_sessions_201707*`

--WHERE totals.visits  1

"""

inspect(query) 

# so all sessions are visits, as it should be 
howto_query7 = """SELECT

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

hitNumber;

        """

view_seqs = retrieve(howto_query7)
print(view_seqs.shape)

view_seqs.head()
# hitNumber has the order within the session

# view_seqs.groupby('fullVisitorId')['visitNumber'].count() # find a sample customer with more than 1 visit

view_seqs[view_seqs['fullVisitorId'] == '0000436683523507380']
# apply list to each group - a way to aggregate the different records into the same row

# nice to see but not recommended to store a list in a dataframe

view_seqs.groupby(['fullVisitorId', 'visitId']).agg({'hitNumber': list, 'pagePath': list}).head(5)
# Get first page from each session - could be nice to use for first touch attribution  

# view_seqs.groupby(['fullVisitorId', 'visitId']).head(1)

# Top 10 landing pages from sessions - homepage 50% of time

view_seqs.groupby(['fullVisitorId', 'visitId']).head(1)['pagePath'].value_counts(normalize=True).head(10)
# expand traffic source 

query = """SELECT

fullVisitorId,

visitId,

visitNumber,

hits.hitNumber AS hitNumber,

trafficSource.*

FROM

`bigquery-public-data.google_analytics_sample.ga_sessions_*`,

UNNEST(hits) as hits

WHERE

_TABLE_SUFFIX BETWEEN '20170701' AND '20170731'

;

        """

inspect(query)

# What are possible sources? in July 2017 value_counts

query = """SELECT

trafficSource.source, COUNT(*) as cnt

FROM

`bigquery-public-data.google_analytics_sample.ga_sessions_201707*`

GROUP BY trafficSource.source

ORDER BY cnt DESC

        """

traffic_sources = retrieve(query)
print(traffic_sources.shape)

print("Too many to plot, even if establishing a threshold at like 100 for July. Need a good way of combining sources")

traffic_sources.head(15)
# confirm (direct) is the source of the (none)'s

query = """SELECT

 trafficSource.source, trafficSource.medium, count(*) cnt

FROM

`bigquery-public-data.google_analytics_sample.ga_sessions_201707*`

WHERE -- trafficSource.source IN ('google', '(direct)') AND 

 trafficSource.medium = '(none)'

GROUP BY 1,2

ORDER BY cnt DESC



        """

inspect(query)
# What are possible media? in 2017 value_counts

query = """SELECT

trafficSource.medium, COUNT(*) as cnt

FROM

`bigquery-public-data.google_analytics_sample.ga_sessions_2017*`

GROUP BY trafficSource.medium

ORDER BY cnt DESC

        """

inspect(query)

# cpc: cost per click - use when has good offer with high conversion rate using ads adapted to that specific offer

# cpm: cost per thousand impression - may save money if good CTR, can buy traffic from premium spots

# cpv = cost per unique visitor
# What are possible devices? value_counts

query = """SELECT

device.browser, COUNT(*) as cnt

FROM

`bigquery-public-data.google_analytics_sample.ga_sessions_201707*`

GROUP BY device.browser

ORDER BY cnt DESC

        """

inspect(query)

# What are possible device categories? value_counts

query = """SELECT

device.deviceCategory, COUNT(*) as cnt

FROM

`bigquery-public-data.google_analytics_sample.ga_sessions_201707*`

GROUP BY device.deviceCategory

ORDER BY cnt DESC

        """

inspect(query)
# What are possible browsers? value_counts -- counts unique users -  but session is more relevant

query = """

SELECT

 deviceCategory, 

 COUNT(*) as cnt

FROM

    (SELECT 

      fullVisitorId,

      visitId,

      MAX(device.deviceCategory) as deviceCategory

    FROM

     `bigquery-public-data.google_analytics_sample.ga_sessions_201707*`

    GROUP BY fullVisitorId, visitId

    ) s

GROUP BY deviceCategory

ORDER BY cnt DESC

        """

inspect(query)
query = """

SELECT  

 totals.sessionQualityDim,

 COUNT(*) cnt

FROM 

`bigquery-public-data.google_analytics_sample.ga_sessions_201707*`

GROUP BY totals.sessionQualityDim

ORDER by cnt DESC



"""

sess_quality_vc = retrieve(query)
import seaborn as sns

sns.set()

sns.scatterplot(x='sessionQualityDim', y='cnt', data=sess_quality_vc[sess_quality_vc['sessionQualityDim'] > 2])

# exclude the two 
sess_quality_vc.sort_values('sessionQualityDim').tail(10)
query = """

SELECT  

 totals.timeOnSite,

 COUNT(*) cnt

FROM 

`bigquery-public-data.google_analytics_sample.ga_sessions_2017*`

GROUP BY totals.timeOnSite

ORDER by cnt DESC



"""

time_on_site = retrieve(query) 

#  totals.timeOnScreen - not a valid column

# timeOnSite: Total time of the session expressed in seconds.
time_on_site['timeOnSite'].describe()
# Base conversion rate in July

query = f"""

SELECT 

    COUNT(totals.totalTransactionRevenue) AS cnt_revenue,

    COUNT(totals.transactions) AS cnt_transactions, 

    COUNT(*) AS cnt_all_rows, --sessions with browser

    COUNT(totals.transactions) / COUNT(*) AS sess_conversion_rate,

    sum(totals.totalTransactionRevenue) AS sum_revenue, 

    Sum(totals.transactions) AS  sum_transactions,

    sum(totals.totalTransactionRevenue) / Sum(totals.transactions) AS revenue_per_transaction

FROM `bigquery-public-data.google_analytics_sample.ga_sessions_201707*` 

"""



base_CR = retrieve(query)
base_CR
# Base conversion rate by month 

query = f"""

SELECT 

    substr(date, 1, 6) as ym,

    COUNT(totals.totalTransactionRevenue) AS cnt_revenue,

    COUNT(totals.transactions) AS cnt_transactions, 

    COUNT(*) AS cnt_all_rows, --sessions with browser

    COUNT(totals.transactions) / COUNT(*) AS sess_conversion_rate,

    sum(totals.totalTransactionRevenue) AS sum_revenue, 

    Sum(totals.transactions) AS  sum_transactions,

    sum(totals.totalTransactionRevenue) / Sum(totals.transactions) AS revenue_per_transaction

FROM `bigquery-public-data.google_analytics_sample.ga_sessions*` 

GROUP BY ym

ORDER BY ym

"""



monthly_base_CR = retrieve(query)
monthly_base_CR

# Not all transactionshave revenue? 
monthly_base_CR['ym'] = pd.to_datetime(monthly_base_CR['ym'], format='%Y%m')
sns.lineplot(x='ym', y='sess_conversion_rate', data=monthly_base_CR)
sns.lineplot(x='ym', y='cnt_all_rows', data=monthly_base_CR)

# the big CR dip in Nov is due more to higher traffic than a drop in transactions
# Base conversion rate by month and device category

query = """

SELECT 

    substr(date, 1, 6) as ym,

    device.deviceCategory,

    COUNT(totals.totalTransactionRevenue) AS cnt_revenue,

    COUNT(totals.transactions) AS cnt_transactions, 

    COUNT(*) AS cnt_all_rows, --sessions with browser

    COUNT(totals.transactions) / COUNT(*) AS sess_conversion_rate,

    sum(totals.totalTransactionRevenue) AS sum_revenue, 

    Sum(totals.transactions) AS  sum_transactions,

    sum(totals.totalTransactionRevenue) / Sum(totals.transactions) AS revenue_per_transaction

FROM `bigquery-public-data.google_analytics_sample.ga_sessions_*`

GROUP BY  device.deviceCategory, ym

ORDER BY device.deviceCategory, ym

"""

monthly_device_CR = retrieve(query)

monthly_device_CR.head()
import matplotlib.pyplot as plt

def plot_metric_by_month(vc_df, metric, group):

    '''Convert date from string and plot metric by group

    vc_df: df, SQL value counts output grouped by month and group

    metric: str, column name for metric of interest

    group: str, column name of group'''

    df = vc_df.copy()

    df['ym'] = pd.to_datetime( df['ym'], format='%Y%m')

    sns.lineplot(x='ym', y=metric, hue=group, data=df, marker='o')

    plt.title(f'Monthly {metric}')

    plt.legend(loc="upper left", bbox_to_anchor=(1,1))
plot_metric_by_month(monthly_device_CR, 'sess_conversion_rate', 'deviceCategory')

# much higher CR when visited via desktop
plot_metric_by_month(monthly_device_CR, 'cnt_transactions', 'deviceCategory')

# raw count looks similar in shape for desktop and mobile, so likely consistent total traffic but changing number of transactions
# Base conversion rate by month and traffic source

query = """

SELECT 

    substr(date, 1, 6) as ym,

    trafficSource.medium, 

    COUNT(totals.totalTransactionRevenue) AS cnt_revenue,

    COUNT(totals.transactions) AS cnt_transactions, 

    COUNT(*) AS cnt_all_rows, --sessions with browser

    COUNT(totals.transactions) / COUNT(*) AS sess_conversion_rate,

    sum(totals.totalTransactionRevenue) AS sum_revenue, 

    Sum(totals.transactions) AS  sum_transactions,

    sum(totals.totalTransactionRevenue) / Sum(totals.transactions) AS revenue_per_transaction

FROM `bigquery-public-data.google_analytics_sample.ga_sessions_*`

GROUP BY trafficSource.medium, ym

ORDER BY trafficSource.medium, ym

"""

monthly_medium_CR = retrieve(query)
plot_metric_by_month(monthly_medium_CR, 'sess_conversion_rate', 'medium')

# direct ads (cpc and cpm) > organic > referral/affiliate

# medium can have a pretty strong effect on CR - diverging from the overall average of 1.5% 
plot_metric_by_month(monthly_medium_CR, 'cnt_transactions', 'medium')

# organic has a low CR but it makes up for a good chunk of the transactions

# Need to look further into (none) --> direct traffic - so people who want to buy stuff often visit directly 

# - bookmarks? returning customers?
# Base conversion rate by month and browser 

query = """

SELECT 

    substr(date, 1, 6) as ym,

    device.browser,

    COUNT(totals.totalTransactionRevenue) AS cnt_revenue,

    COUNT(totals.transactions) AS cnt_transactions, 

    COUNT(*) AS cnt_all_rows, --sessions with browser

    COUNT(totals.transactions) / COUNT(*) AS sess_conversion_rate,

    sum(totals.totalTransactionRevenue) AS sum_revenue, 

    Sum(totals.transactions) AS  sum_transactions,

    sum(totals.totalTransactionRevenue) / Sum(totals.transactions) AS revenue_per_transaction

FROM `bigquery-public-data.google_analytics_sample.ga_sessions_*`

GROUP BY  device.browser, ym

ORDER BY  device.browser, ym

"""

monthly_browser_CR = retrieve(query)
mask = (monthly_browser_CR['sess_conversion_rate'] > 0) | (monthly_browser_CR['cnt_all_rows'] > 1000) # doesn't catch every month if a browser doesn't always make it

print(len(monthly_browser_CR[mask]))

monthly_browser_CR[mask].sample(10)
plot_metric_by_month(monthly_browser_CR[mask], 'sess_conversion_rate', 'browser')

# Chrome dominates both raw count and CR

# Silk has that one blip of relatively high CR
plot_metric_by_month(monthly_browser_CR[mask], 'cnt_transactions', 'browser')

plot_metric_by_month(monthly_browser_CR[mask & (monthly_browser_CR['browser'] != 'Chrome')], 'cnt_transactions', 'browser')

# Safari being a distant second but another sizeable gap from the rest


# Base conversion rate by month and source

query = """

SELECT 

    substr(date, 1, 6) as ym,

    trafficSource.source,

    COUNT(totals.totalTransactionRevenue) AS cnt_revenue,

    COUNT(totals.transactions) AS cnt_transactions, 

    COUNT(*) AS cnt_all_rows, --sessions with browser

    COUNT(totals.transactions) / COUNT(*) AS sess_conversion_rate,

    sum(totals.totalTransactionRevenue) AS sum_revenue, 

    Sum(totals.transactions) AS  sum_transactions,

    sum(totals.totalTransactionRevenue) / Sum(totals.transactions) AS revenue_per_transaction

FROM `bigquery-public-data.google_analytics_sample.ga_sessions_*`

GROUP BY trafficSource.source, ym

ORDER BY  trafficSource.source, ym

"""

monthly_source_CR = retrieve(query)
monthly_source_CR.head(10)
monthly_source_CR.shape
mask = (monthly_source_CR['cnt_all_rows'] > 500) 

plot_metric_by_month(monthly_source_CR[mask], 'sess_conversion_rate', 'source')

# dominated by direct or google

# not sure what dfa is
mask = (monthly_source_CR['cnt_all_rows'] > 500) 

plot_metric_by_month(monthly_source_CR[mask], 'sess_conversion_rate', 'source')
# looking into a particuar visitor

query = """SELECT date, visitId, channelGrouping, visitNumber, trafficSource.medium, 

            totals.*

            FROM `bigquery-public-data.google_analytics_sample.ga_sessions_*`

            WHERE  fullVisitorId =  '7813149961404844386'

            ORDER BY date 

            

            """

q = retrieve(query)



q.head()
# channelgrouping - ga channels

query = """SELECT channelGrouping, COUNT(*) cnt

            FROM `bigquery-public-data.google_analytics_sample.ga_sessions_*`

            GROUP BY channelGrouping

            """

inspect(query)
query = """SELECT channelGrouping, COUNT(DISTINCT trafficSource.medium) cnt

            FROM `bigquery-public-data.google_analytics_sample.ga_sessions_*`

            GROUP BY channelGrouping

            """

inspect(query)
# verify conditions match up with definition  https://support.google.com/analytics/answer/3297892

query = """SELECT channelGrouping, array_agg(DISTINCT trafficSource.medium) cnt

            FROM `bigquery-public-data.google_analytics_sample.ga_sessions_*`

            GROUP BY channelGrouping

            """

inspect(query)
# above by with counts within each group

query = """SELECT channelGrouping,trafficSource.medium, count(*) cnt

            FROM `bigquery-public-data.google_analytics_sample.ga_sessions_*`

            GROUP BY channelGrouping,trafficSource.medium

            """

inspect(query)
# group by medium

query = """SELECT trafficSource.medium,  channelGrouping, count(*) cnt

            FROM `bigquery-public-data.google_analytics_sample.ga_sessions_*`

            GROUP BY trafficSource.medium, channelGrouping

            ORDER BY trafficSource.medium

            """

inspect(query)

# so (none) can have labeled in channelGroupig even when 
# what do these look like? 

query = """SELECT * 

            FROM `bigquery-public-data.google_analytics_sample.ga_sessions_*`

            WHERE trafficSource.medium = '(none)'

                AND channelGrouping = 'Referral'

            

            """

inspect(query)
query = """

SELECT 

    substr(date, 1, 6) as ym,

    channelGrouping,

    COUNT(totals.totalTransactionRevenue) AS cnt_revenue,

    COUNT(totals.transactions) AS cnt_transactions, 

    COUNT(*) AS cnt_all_rows, --sessions with browser

    COUNT(totals.transactions) / COUNT(*) AS sess_conversion_rate,

    sum(totals.totalTransactionRevenue) AS sum_revenue, 

    Sum(totals.transactions) AS  sum_transactions,

    sum(totals.totalTransactionRevenue) / Sum(totals.transactions) AS revenue_per_transaction

FROM `bigquery-public-data.google_analytics_sample.ga_sessions_*`

GROUP BY  channelGrouping, ym

ORDER BY channelGrouping, ym

"""

monthly_channel_CR = retrieve(query)

monthly_channel_CR.head()
monthly_channel_CR.sum()
plot_metric_by_month(monthly_channel_CR, 'sess_conversion_rate', 'channelGrouping')
plot_metric_by_month(monthly_channel_CR, 'cnt_transactions', 'channelGrouping')

# Why does this look so different than raw source medium

    # because (none) can be assigned to various channelGroupings and other caveats

# https://support.google.com/analytics/answer/3297892
# Base conversion rate by month and traffic source

query = """

SELECT 

    substr(date, 1, 6) as ym,

    trafficSource.medium, 

    COUNT(totals.totalTransactionRevenue) AS cnt_revenue,

    COUNT(totals.transactions) AS cnt_transactions, 

    COUNT(*) AS cnt_all_rows, --sessions with browser

    COUNT(totals.transactions) / COUNT(*) AS sess_conversion_rate,

    sum(totals.totalTransactionRevenue) AS sum_revenue, 

    Sum(totals.transactions) AS  sum_transactions,

    sum(totals.totalTransactionRevenue) / Sum(totals.transactions) AS revenue_per_transaction

FROM `bigquery-public-data.google_analytics_sample.ga_sessions_*`

GROUP BY trafficSource.medium, ym

ORDER BY trafficSource.medium, ym

"""

monthly_medium_CR = retrieve(query)
monthly_medium_CR.sum()

# revenue per transaction and sess_conversion_rate changed -- rest stayed the same
plot_metric_by_month(monthly_medium_CR, 'sess_conversion_rate', 'medium')

# direct ads (cpc and cpm) > organic > referral/affiliate

# medium can have a pretty strong effect on CR - diverging from the overall average of 1.5% 
plot_metric_by_month(monthly_medium_CR, 'cnt_transactions', 'medium')

# organic has a low CR but it makes up for a good chunk of the transactions

# Need to look further into (none) --> direct traffic - so people who want to buy stuff often visit directly 

# - bookmarks? returning customers?