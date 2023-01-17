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
# See https://www.kaggle.com/dansbecker/getting-started-with-sql-and-bigquery
air_pollution = bq_helper.BigQueryHelper("bigquery-public-data", "epa_historical_air_quality")
air_pollution.list_tables()
# print information on all the columns in the "full" table
# in the hacker_news dataset
air_pollution.table_schema("air_quality_annual_summary")
# check the first couple of lines to make sure the data is correct
air_pollution.head("air_quality_annual_summary")
# .head() method could also be used to check a specific column
air_pollution.head("air_quality_annual_summary", selected_columns = 'ninety_five_percentile', num_rows = 12)
# Estimate the query size of a dataset
query = """SELECT state_code
            FROM `bigquery-public-data.epa_historical_air_quality.air_quality_annual_summary`
            WHERE county_code = '001'
        """ 

air_pollution.estimate_query_size(query)

# Run the query
air_pollution.query_to_pandas_safe(query, max_gb_scanned=0.02)
# Play around with the query
states = air_pollution.query_to_pandas(query)
print(int(states.max()))
import matplotlib.pyplot as plt
plt.figure(1,(12,3))
plt.hist(states,edgecolor='black', linewidth=1.2, bins = int(states.max()))

plt.show()
states.state_code.value_counts().head()
# try GROUP BY and HAVING methods

query = """
        SELECT arithmetic_mean, COUNT(state_code)
        FROM `bigquery-public-data.epa_historical_air_quality.air_quality_annual_summary`
        GROUP BY arithmetic_mean
        HAVING COUNT(state_code) > 10
"""
mean = air_pollution.query_to_pandas_safe(query)
print(mean.head())
query = """
        SELECT arithmetic_mean, state_code, site_num
        FROM `bigquery-public-data.epa_historical_air_quality.air_quality_annual_summary`
        ORDER BY state_code DESC
"""
mean = air_pollution.query_to_pandas_safe(query, max_gb_scanned=0.04)
print(mean.head())