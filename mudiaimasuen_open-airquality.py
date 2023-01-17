# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
# import our bq_helper package
import bq_helper 
# Any results you write to the current directory are saved as output.
# create a helper object for this dataset
open_air_quality = bq_helper.BigQueryHelper(active_project = "bigquery-public-data", dataset_name="openaq")

open_air_quality.list_tables()
open_air_quality.head('global_air_quality', num_rows = 10)
#query to select the countries that use a unit other than ppm to measure pollution
query = """
SELECT DISTINCT country, unit, pollutant
FROM `bigquery-public-data.openaq.global_air_quality`
WHERE unit != 'ppm'
"""
#size of query
open_air_quality.estimate_query_size(query)
open_air_quality_df = open_air_quality.query_to_pandas_safe(query, max_gb_scanned=0.4)
open_air_quality_df
query = """
SELECT location, unit, pollutant, value
FROM `bigquery-public-data.openaq.global_air_quality`
WHERE value = 0.00

"""
#check the size of the query
open_air_quality.estimate_query_size(query)
air_quality_value_df = open_air_quality.query_to_pandas_safe(query, max_gb_scanned = 0.4)
air_quality_value_df
