# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

import bq_helper #making life easier for BigQuery
# Any results you write to the current directory are saved as output.
#creating helper object to store bigquery dataset
traffic_deaths = bq_helper.BigQueryHelper(active_project = "bigquery-public-data", dataset_name = "nhtsa_traffic_fatalities")
#Printing list of tables in nhtsa_traffic_fatalities dataset
traffic_deaths.list_tables()
#print information on all columns in the "accident_2016" table 
#from the nhtsa_traffic_fatalities dataset
traffic_deaths.table_schema("accident_2016")

#Printing the head of the accident_2016 table
traffic_deaths.head("accident_2016")
#Checking a single column within the accident_2016 dataset
traffic_deaths.head("accident_2016", selected_columns = "number_of_fatalities", num_rows=10)
#Checking how big a specific query is
#This query goes through the full table of accident_2016
#and pulls out the minute_of_ems_arrival_at_hospital column for accidents in Georgia

query = """SELECT minute_of_ems_arrival_at_hospital
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2016`
            WHERE state_name = "Georgia" """
#Checking how big the query is 
traffic_deaths.estimate_query_size(query)

#Running the query
hospital_time_georgia = traffic_deaths.query_to_pandas_safe(query)

#average time to hospital after fatal wreck in Georgia
hospital_time_georgia.minute_of_ems_arrival_at_hospital.mean()
