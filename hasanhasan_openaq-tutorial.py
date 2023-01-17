# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# import our bq_helper package
import bq_helper 

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
# print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

#global variabls
globalaq = 'global_air_quality'
# create a helper object for our bigquery dataset
openaq = bq_helper.BigQueryHelper(active_project= "bigquery-public-data", 
                                       dataset_name = "openaq")
# print a list of all the tables in the openaq dataset
openaq.list_tables()
# print information on all the columns in the "global_air_quality" table
# in the openaq dataset
openaq.table_schema('global_air_quality')
# preview the first couple lines of the "global_air_quality" table
openaq.head("global_air_quality")
# preview the first ten entries in the by column of the global_air_quality table
openaq.head("global_air_quality", selected_columns="location", num_rows=10)
# this query looks in the full table in the global air qiality
# dataset, then gets the score column from every row where 
# the country column has "CA" in it
query = """SELECT value
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE country = "CA"
            """

# check how big this query will be
openaq.estimate_query_size(query)
# only run this query if it's less than 100 MB
openaq.query_to_pandas_safe(query, max_gb_scanned=0.1)
# Average canadian air quality value
canada_aq.value.mean()
# save our dataframe as a .csv 
canada_aq.to_csv("canada_aq.csv")
