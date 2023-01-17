# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import bq_helper
accidents2016 = bq_helper.BigQueryHelper(active_project="bigquery-public-data",dataset_name="nhtsa_traffic_fatalities")
accidents2016.list_tables()
#number of accidents each hour arranged in dec order with 1st column mentioning the number of accidents and 2nd one mentioning t
accidents2016.table_schema("accident_2016")
accidents2016.head("accident_2016",num_rows=2)
query = """select count(consecutive_number),
extract(HOUR from timestamp_of_crash)
FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2016`
GROUP BY extract(hour from timestamp_of_crash)
ORDER BY count(consecutive_number) DESC """
accidents2016.estimate_query_size(query)
accidents2016.query_to_pandas_safe(query)
accidents2016.table_schema("vehicle_2016")
accidents2016.head("vehicle_2016",num_rows=2)
accidents2016.head("vehicle_2016",selected_columns="hit_and_run",num_rows=2)
#for arranging the data by the state hit and run case in dec order
query1="""select registration_state_name,count(consecutive_number)
from `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_2016`
where hit_and_run="Yes"
Group by registration_state_name
ORDER BY count(consecutive_number) DESC
"""
accidents2016.query_to_pandas_safe(query1)