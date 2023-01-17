# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import bq_helper # help with BigQuery



# Any results you write to the current directory are saved as output.


nhtsa_fatalities = bq_helper.BigQueryHelper(active_project='bigquery-public-data', dataset_name='nhtsa_traffic_fatalities')
nhtsa_fatalities.list_tables()
nhtsa_fatalities.table_schema('accident_2015')
nhtsa_fatalities.head('accident_2015')
nhtsa_fatalities.head('accident_2015', selected_columns='minute_of_ems_arrival_at_hospital', num_rows=10)
# this query looks in the accident_2015 table in the hacker_news
# dataset, then gets the score column from every row where 
# the type column has "job" in it.
query = """SELECT state_name, number_of_motor_vehicles_in_transport_mvit, number_of_persons_in_motor_vehicles_in_transport_mvit, minute_of_ems_arrival_at_hospital
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`
            WHERE number_of_drunk_drivers = 1 """

# check how big this query will be
nhtsa_fatalities.estimate_query_size(query)
drunk_driver_states_df = nhtsa_fatalities.query_to_pandas_safe(query, max_gb_scanned=0.1)
drunk_driver_states_df
# save our dataframe as a .csv 
drunk_driver_states_df.to_csv("drunk_driver_fatalities.csv")