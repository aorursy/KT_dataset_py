# import package with helper functions 
import bq_helper

# create a helper object for this dataset
open_aq = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="openaq")

# print all the tables in this dataset (there's only one!)
open_aq.list_tables()
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# import package with helper functions 
import bq_helper

# create a helper object for this dataset
open_aq = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="openaq")

# print all the tables in this dataset (there's only one!)
open_aq.list_tables()

# print the first couple rows of the "global_air_quality" dataset
open_aq.head("global_air_quality")

query = """SELECT country
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE pollutant != 'ppm'
        """

# the query_to_pandas_safe will only return a result if it's less
# than one gigabyte (by default)
pollution_unit = open_aq.query_to_pandas_safe(query)

# What five cities have the most measurements taken there?
pollution_unit.country.value_counts().head()
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# import package with helper functions 
import bq_helper

#Create a helper for this dataset
open_aq = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="openaq")

#print all the tables in a dataset
open_aq.list_tables()
open_aq.head("global_air_quality")

query = """SELECT source_name
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE value = 0
        """

#return the query only if it's
#less than GB
pollution_value = open_aq.query_to_pandas_safe(query)
pollution_value.source_name.value_counts().head()
