# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import bq_helper # big query
openaq = bq_helper.BigQueryHelper(active_project='bigquery-public-data', dataset_name='openaq')
openaq.list_tables()
openaq.head('global_air_quality')
query = """ 
SELECT DISTINCT country, unit
FROM `bigquery-public-data.openaq.global_air_quality`
WHERE unit != 'ppm'
"""

# check how big this query will be
openaq.estimate_query_size(query)
aq_df = openaq.query_to_pandas_safe(query, max_gb_scanned=0.1)
aq_df
query = """ 
SELECT location, pollutant
FROM `bigquery-public-data.openaq.global_air_quality`
WHERE value = 0.00
"""

# check how big this query will be
openaq.estimate_query_size(query)
aq_df = openaq.query_to_pandas_safe(query, max_gb_scanned=0.1)
aq_df