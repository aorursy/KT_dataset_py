# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import bq_helper # import package with helper package

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# create helper object for the dataset 
google_analytics = bq_helper.BigQueryHelper(active_project="bigquery-public-data", dataset_name="google_analytics_sample")

# What is the total number of transactions generated per device browser in July 2017?
query = """
        SELECT device.operatingSystem as device, SUM(totals.transactions) as transactions
        FROM `bigquery-public-data.google_analytics_sample.ga_sessions_*`
        WHERE
        _TABLE_SUFFIX BETWEEN '20170701' AND '20170731'
        GROUP BY device
        ORDER BY transactions DESC
        """
transactions = google_analytics.query_to_pandas_safe(query)
transactions
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import bq_helper # import package with helper package

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# create helper object for the dataset 
google_analytics = bq_helper.BigQueryHelper(active_project="bigquery-public-data", dataset_name="google_analytics_sample")

# real bounce rate = percentage of visits with a single page view, how many times did we see a single page visit grouped by traffic source
# real bounce rate = (single page view visits/ total combined page views) * 100
query = """
        SELECT sources, total_pageviews, bounces,
        ((bounces/total_pageviews)*100) AS bounce_rate, totalviews2
        FROM (
        SELECT SUM(totals.bounces) AS bounces, COUNT(totals.pageviews) AS total_pageviews,
        trafficSource.source AS sources, COUNT(trafficSource.source) AS totalviews2
        FROM `bigquery-public-data.google_analytics_sample.ga_sessions_*`
        WHERE
        _TABLE_SUFFIX BETWEEN '20170701' AND '20170731'
        GROUP BY sources
        )
        ORDER BY total_pageviews DESC, bounce_rate, bounces
        """ 
bounce_rate = google_analytics.query_to_pandas_safe(query)
bounce_rate.head(20)

# What was the average number of product pageviews for users who made a purchase in July 2017?
# What was the average number of product pageviews for users who did not make a purchase in July 2017?
# What was the average total transactions per user that made a purchase in July 2017?
# What is the average amount of money spent per session in July 2017?
# What is the sequence of pages viewed?
# What was the average number of product pageviews for users who made a purchase in July 2017?
query2 = """SELECT visitID, totals.transactionRevenue AS revenue, totals.pageviews AS pageviews
            FROM `bigquery-public-data.google_analytics_sample.ga_sessions_*`
            WHERE 
            (_TABLE_SUFFIX BETWEEN '20170701' AND '20170731') AND totals.transactionRevenue > 0
            AND visitID = 1500404409
            ORDER BY revenue DESC, visitID, pageviews
         """

query3 = """SELECT AVG(totals.pageviews) AS pageviews, totals.transactions AS transactions
            FROM `bigquery-public-data.google_analytics_sample.ga_sessions_*`
            WHERE 
            (_TABLE_SUFFIX BETWEEN '20170701' AND '20170731')
            GROUP BY transactions
         """
# total page views /  users who made a purchase. 
response2 = google_analytics.query_to_pandas_safe(query3)
response2.head(100)