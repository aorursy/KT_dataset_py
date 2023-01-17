# SQL Scavenger Hunt: Day 1
# Reference from: https://www.kaggle.com/rtatman/sql-scavenger-hunt-day-1/

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
# import package with helper functions 
import bq_helper

# create a helper object for this dataset
open_aq = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                  dataset_name="openaq")

# print all the tables in this dataset (there's only one!)
open_aq.list_tables()
open_aq.head("global_air_quality")
# test query with WHERE
query = """
SELECT city, country
FROM `bigquery-public-data.openaq.global_air_quality`
WHERE country='US'
"""
test_dataset = open_aq.query_to_pandas_safe(query=query)
test_dataset.head()
test_dataset.city.value_counts().head()
# 1
query_non_ppm = """
SELECT country, unit
FROM `bigquery-public-data.openaq.global_air_quality`
WHERE unit!='ppm'
"""
dataset_non_ppm = open_aq.query_to_pandas_safe(query=query_non_ppm)
dataset_non_ppm.head()
dataset_non_ppm.unit.value_counts()
# 2
query_pollutant_0 = """
SELECT pollutant
FROM `bigquery-public-data.openaq.global_air_quality`
WHERE value=0.0
"""
dataset_pollutant_0 = open_aq.query_to_pandas_safe(query=query_pollutant_0)
dataset_pollutant_0.head()