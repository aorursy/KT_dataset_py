# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
#import BigQuery Helper
import bq_helper
#create a helper variable for dataset
open_aq = bq_helper.BigQueryHelper(active_project = "bigquery-public-data", dataset_name ="openaq")

#print all the tables in this dataset
open_aq.list_tables()
# Question 1: What countries use a unit other than ppm to measure any type of pollution?
query2 = """ SELECT country 
            FROM `bigquery-public-data.openaq.global_air_quality` 
            WHERE unit != 'ppm' 
        """
# Set up non PPM dataframe
non_ppm = open_aq.query_to_pandas_safe(query2)
# display query results
non_ppm.country.value_counts()
# query to find pollutants that have a value of exactly 0
query3 = """ SELECT pollutant
              FROM `bigquery-public-data.openaq.global_air_quality`
              WHERE value = 0
        """
#assign variable to pollutant value of 0
null_poll = open_aq.query_to_pandas_safe(query3)
null_poll.pollutant.value_counts()