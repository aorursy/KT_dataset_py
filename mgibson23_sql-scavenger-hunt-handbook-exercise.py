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
# import our bq_helper package
import bq_helper 

# create a helper object for our bigquery dataset
air_quality = bq_helper.BigQueryHelper(active_project= "bigquery-public-data", 
                                       dataset_name = "epa_historical_air_quality")

# print a list of all the tables in the hacker_news dataset
air_quality.list_tables()
# print information on all the columns in the "air_quality_annual_summary" table
# in the air_quality dataset
air_quality.table_schema("air_quality_annual_summary")


air_quality.head("air_quality_annual_summary")
# preview the first ten entries in the by column of the full table
air_quality.head("air_quality_annual_summary", selected_columns="state_code", num_rows=10)
# this query looks in the air_quality_annual_summary table in the air_quality
# dataset, then gets the latitude column from every row where 
# the state_code column has "16" in it.
query = """SELECT latitude
            FROM `bigquery-public-data.epa_historical_air_quality.air_quality_annual_summary`
            WHERE state_code = "16" """

# check how big this query will be in GB
air_quality.estimate_query_size(query)
# only run this query if it's less than 100 MB
air_quality.query_to_pandas_safe(query, max_gb_scanned=0.1)
# check out the latitudes (if the 
# query is smaller than 1 gig)
state_latitude = air_quality.query_to_pandas_safe(query)

# average latitudes for state
state_latitude.mean()


# save the dataframe as a .csv 
state_latitude.to_csv("state_latitude.csv")

