# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
# import package with helper functions 
import bq_helper
# create a helper object for this dataset
open_aq = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="openaq")

# print all the tables in this dataset (there's only one!)
open_aq.list_tables()
# query to select all the items from the "city" column where the
# "country" column is "us"
query = """SELECT city
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE country = 'US'
        """
# the query_to_pandas_safe will only return a result if it's less
# than one gigabyte (by default)
us_cities = open_aq.query_to_pandas_safe(query)
us_cities.city.value_counts().head()
query2="""SELECT country
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE unit!='ppm'
            """
countries_without_ppm =open_aq.query_to_pandas_safe(query2)
print(len(countries_without_ppm.country.unique()))
query3="""SELECT pollutant
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE value=0
           """
polutatnts_with0 =open_aq.query_to_pandas_safe(query3)
polutatnts_with0.pollutant.unique()
