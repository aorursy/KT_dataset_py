# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import bq_helper

open_aq = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="openaq")
open_aq.list_tables()
open_aq.head('global_air_quality')
query = """SELECT city 
           FROM `bigquery-public-data.openaq.global_air_quality`
           WHERE country = 'US' 
        """
# the argument we pass to FROM is not in single or double quotation marks (' or "). 
# It is in backticks (`)
# the query_to_pandas_safe will only return a result if it's less
# than one gigabyte (by default)
# so it is safe to run it because it won't run a query if it's larger than 1 gigabyte
us_cities = open_aq.query_to_pandas_safe(query)
us_cities.head(30)
