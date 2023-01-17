# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import bq_helper # BigQuery helper

nhtsa_traffic_fatalities = bq_helper.BigQueryHelper(active_project='bigquery-public-data', dataset_name='nhtsa_traffic_fatalities')
nhtsa_traffic_fatalities.head('accident_2015')
query = """
SELECT COUNT('consecutive_number'), FORMAT_TIME('%H', TIME(timestamp_of_crash))
FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`
GROUP BY FORMAT_TIME('%H', TIME(timestamp_of_crash))
ORDER BY COUNT(consecutive_number) DESC
"""

# check how big this query will be
nhtsa_traffic_fatalities.estimate_query_size(query)
accidents_by_hour = nhtsa_traffic_fatalities.query_to_pandas_safe(query, max_gb_scanned=0.5)
accidents_by_hour.head()
query = """
SELECT registration_state_name, COUNT(registration_state_name)
FROM `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_2015`
where hit_and_run LIKE '%Y%'
GROUP BY registration_state_name
ORDER BY COUNT(registration_state_name) DESC
"""

# check how big this query will be
nhtsa_traffic_fatalities.estimate_query_size(query)
hit_and_run = nhtsa_traffic_fatalities.query_to_pandas_safe(query, max_gb_scanned=0.5)
hit_and_run.head()