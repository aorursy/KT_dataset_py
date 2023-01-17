# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import bq_helper 
OpenAQ = bq_helper.BigQueryHelper(active_project= "bigquery-public-data", 
                                       dataset_name = "openaq")

OpenAQ.head("global_air_quality")
# Any results you write to the current directory are saved as output.
# question 1
# Which countries use a unit other than ppm to measure any type of pollution? (Hint: to get rows where the value *isn't* something, use "!=")

query_1 = """SELECT DISTINCT(country) FROM `bigquery-public-data.openaq.global_air_quality` WHERE unit != 'ppm'"""
print('query size = ', OpenAQ.estimate_query_size(query_1))
print(OpenAQ.query_to_pandas_safe(query_1))


# Question 2 
# Which pollutants have a value of exactly 0?
query_2 = """SELECT DISTINCT(pollutant) FROM `bigquery-public-data.openaq.global_air_quality` WHERE value = 0"""
print('query 2 result size = ', OpenAQ.estimate_query_size(query_2))
print(OpenAQ.query_to_pandas_safe(query_2))
