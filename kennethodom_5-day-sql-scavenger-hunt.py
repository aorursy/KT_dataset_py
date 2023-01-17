# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
import bq_helper
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

openaq = bq_helper.BigQueryHelper(active_project="bigquery-public-data", dataset_name="openaq")
openaq.list_tables()
openaq.table_schema("global_air_quality")
openaq.head("global_air_quality")
openaq.head("global_air_quality", selected_columns="unit", num_rows=20)
query = """SELECT country
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE unit != "ppm" """

countries_not_using_ppm = openaq.query_to_pandas_safe(query)
countries_not_using_ppm
query1 = """SELECT pollutant
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE value = 0 """

pollutants_with_zero = openaq.query_to_pandas_safe(query1)
pollutants_with_zero
