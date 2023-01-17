# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


import bq_helper # import our bq_helper package
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

# Any results you write to the current directory are saved as output.
traffic_fatality = bq_helper.BigQueryHelper(active_project = "bigquery-public-data", dataset_name ="nhtsa_traffic_fatalities" )

###Print a list of all the tables
#getting the list of tables in the database
traffic_fatality.list_tables()

#Number of table in the database
print("This database has " + str(len(traffic_fatality.list_tables())) + " Tables")
traffic_fatality.table_schema("accident_2015")
# preview the first couple lines of the "accident_2015" table
traffic_fatality.head("accident_2015")
traffic_fatality.head("accident_2015", selected_columns="timestamp_of_crash", num_rows=10)
# this query looks in the accident_2015 table in the traffic_fatality
# dataset, then gets the time_of_crash and state_name column from every row where 
# where number_of_motor_vehicles_in_transport_mvit > 5
query ="""SELECT state_name, timestamp_of_crash, number_of_motor_vehicles_in_transport_mvit
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`
            WHERE number_of_motor_vehicles_in_transport_mvit >= 5"""

# check how big this query will be
traffic_fatality.estimate_query_size(query)
sates_with_fatal_crashes_equal_to_or_above_5_cars_df = traffic_fatality.query_to_pandas_safe(query, max_gb_scanned=0.2)
sates_with_fatal_crashes_equal_to_or_above_5_cars_df
#this saves our dataframe as a .csv
sates_with_fatal_crashes_equal_to_or_above_5_cars_df.to_csv("states_with_fatalities_involving_above_4_automobiles")
