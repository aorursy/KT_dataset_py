# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import bq_helper

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
# print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
# Create a BQ Helper object
air_quality = bq_helper.BigQueryHelper(active_project = "bigquery-public-data",
                               dataset_name = "openaq")
# List of tables in the dataset
air_quality.list_tables()
# List the columns and their definitions in the table
air_quality.table_schema("global_air_quality")
# Let's see a few rows in the given table!
air_quality.head("global_air_quality")
# sample query to test the estimate_size helper
# This query finds the average of ozone pollution in the cities that are affected by it in the US.
# It is sorted in descending order by the average value.
query = """SELECT city, avg(value) as o3_value
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE pollutant = "o3" and country = "US" 
            group by city
            order by o3_value desc"""
print("Query size = ", round(air_quality.estimate_query_size(query) * 1000,3), "MB")
# Store the query results in a pandas dataframe
us_o3_values = air_quality.query_to_pandas_safe(query)
# Seems like Chattanooga in Tennessee has an ozone problem!
# You might even find some news articles dating back a few years talking about this problem.
us_o3_values.head()
# Now, coming to the scavenger hunt questions...
# Which countries use a unit other than ppm to measure any type of pollution? 

query1 = """SELECT country
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE
            unit != "ppm"
            group by country
        """
print("Query size = ", round(air_quality.estimate_query_size(query1) * 1000,3), "MB")
# Store the result in a pandas dataframe
countries_using_other = air_quality.query_to_pandas_safe(query1)
# Let's see a few of the countries, shall we?
countries_using_other.head()
# 64 countries are using a unit other than ppm to measure pollutants
countries_using_other.describe()
# Which pollutants have a value of exactly 0?

query2 = """SELECT pollutant
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE
            value = 0
            group by pollutant
        """
print("Query size = ", round(air_quality.estimate_query_size(query1) * 1000,3), "MB")
# Store the result in a pandas dataframe
pollutants_with_zero_value = air_quality.query_to_pandas_safe(query2)
# All of the pollutants seem to have a value of exactly zero in some row
pollutants_with_zero_value
