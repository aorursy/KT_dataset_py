# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import bq_helper

#Reading the table
accident = bq_helper.BigQueryHelper(active_project='bigquery-public-data',dataset_name='nhtsa_traffic_fatalities')
# Glimpse of the table:
accident.list_tables()
# For this the hint has been provided as to extract from the accident_2016 or accident_2015 table.
accident.table_schema('accident_2016')
accident.head('accident_2016')
query = """SELECT COUNT(consecutive_number), 
                  EXTRACT(HOUR FROM timestamp_of_crash)
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2016`
            GROUP BY EXTRACT(HOUR FROM timestamp_of_crash)
            ORDER BY COUNT(consecutive_number) DESC
        """
accident.estimate_query_size(query)
accident.query_to_pandas_safe(query)
accident.head('vehicle_2016')
query= """SELECT state_number,COUNT(consecutive_number) from `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_2016` group by state_number order by count(consecutive_number) DESC """
accident.estimate_query_size(query)
accident.query_to_pandas_safe(query)
