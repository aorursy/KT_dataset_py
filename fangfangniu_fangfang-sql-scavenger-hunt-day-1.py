# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
# import the bq_helper package
import bq_helper 
# create a helper object for the bigquery dataset
openaq = bq_helper.BigQueryHelper(active_project= "bigquery-public-data", 
                                       dataset_name = "openaq")
# print a list of all the tables in the openaq dataset
openaq.list_tables()
# print information on all the columns in the "global_air_quality" table
# in the openaq dataset
openaq.table_schema("global_air_quality")
# preview the first couple lines of the "global_air_quality" table
openaq.head("global_air_quality")
# preview the first twenty entries in the unit column of the global_air_quality table
openaq.head("global_air_quality", selected_columns="unit", num_rows=20)
# this query looks in the global_air_quality table in the openaq
# dataset, then gets the country column from every row where 
# the unit to measure any type of pollution is not ppm.
query = """SELECT country
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE unit != "ppm" """

# check how big this query will be
openaq.estimate_query_size(query)
# actually run 'countries not using ppm' the query
countries_not_ppm = openaq.query_to_pandas_safe(query, max_gb_scanned=0.1)
# value counts for countries which use a unit other than ppm to measure any type of pollution
countries_not_ppm.country.value_counts().head(10)
# total number of countries which use a unit other than ppm to measure any type of pollution
countries_not_ppm.country.value_counts().count()
# preview the first twenty entries in the value column of the global_air_quality table
openaq.head("global_air_quality", selected_columns="value", num_rows=20)
# this query looks in the global_air_quality table in the openaq
# dataset, then gets the pollutant column from every row where 
# the value column has "0.000" in it.
query = """SELECT pollutant
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE value = 0.000 """

# check how big this query will be
openaq.estimate_query_size(query)
# actually run the 'pollutant value' query
pollutant_zero = openaq.query_to_pandas_safe(query, max_gb_scanned=0.1)
pollutant_zero.pollutant.value_counts()