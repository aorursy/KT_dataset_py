# import package with helper functions 
import bq_helper

# create a helper object for this dataset
open_aq = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="openaq")

# print all the tables in this dataset (there's only one!)
open_aq.list_tables()
# print the first couple rows of the "global_air_quality" dataset
open_aq.head("global_air_quality")
# query to select all the items from the "city" column where the
# "country" column is "us"
query = """SELECT city
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE country = 'US'
        """
# the query_to_pandas_safe will only return a result if it's less
# than one gigabyte (by default)
us_cities = open_aq.query_to_pandas_safe(query)
# What five cities have the most measurements taken there?
us_cities.city.value_counts().head()
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output

# Any results you write to the current directory are saved as output.
import bq_helper 

# import package with helper functions 
import bq_helper

# create a helper object for this dataset
open_aq = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="openaq")


# Which countries use a unit other than ppm to measure any type of pollution?

query2=  """select distinct country
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE unit!='ppm'
         """
countries=open_aq.query_to_pandas_safe(query2)
countries
# Which pollutants have a value of exactly 0?

# Anytime anywhere
query3=  """select  pollutant
            FROM `bigquery-public-data.openaq.global_air_quality` 
            where value=0
            group by pollutant
            """
pollutants=open_aq.query_to_pandas_safe(query3)
pollutants

#Which pollutants have a value of exactly 0? (Today)
query4=  """select  pollutant
            FROM `bigquery-public-data.openaq.global_air_quality`
            where (EXTRACT(DATE FROM timestamp))=CURRENT_DATE() and value=0
            group by pollutant
            """

pollutants2=open_aq.query_to_pandas_safe(query4)
pollutants2

