# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import bq_helper

from bq_helper import BigQueryHelper





# Establish Helper Object for data scanning

google_analytics = bq_helper.BigQueryHelper(active_project="bigquery-public-data",

                                   dataset_name="google_analytics_sample")



bq_assistant = BigQueryHelper("bigquery-public-data", "google_analytics_sample")

bq_assistant.list_tables()



tablelist = google_analytics.list_tables()

print(tablelist)





query_everything = """

#standardSQL

SELECT *

FROM 

    # enclose table names with WILDCARDS in backticks `` , not quotes ''

    `bigquery-public-data.google_analytics_sample.ga_sessions_*`

WHERE

    _TABLE_SUFFIX BETWEEN '20161130' AND '20170101'

"""

print(google_analytics.estimate_query_size(query_everything))

oneTable = google_analytics.query_to_pandas_safe(query_everything, max_gb_scanned=2)

oneTable.to_csv(r'ga_sessions_201612.csv')











# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.