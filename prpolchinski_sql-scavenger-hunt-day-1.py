# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# import our bq_helper package
import bq_helper 

# create a helper object for our bigquery dataset
Data_day1 = bq_helper.BigQueryHelper(active_project= "bigquery-public-data", 
                                       dataset_name = "openaq")

# print a list of all the tables in the hacker_news dataset
Data_day1.list_tables()

# print information on all the columns in the "full" table
# in the hacker_news dataset
Data_day1.table_schema("global_air_quality")

# preview the first couple lines of the "full" table
Data_day1.head("global_air_quality")


# Query to list different values of measurement unit
query1 = """SELECT distinct unit, count(*) as UnitCount
            FROM `bigquery-public-data.openaq.global_air_quality`
            GROUP BY unit """

# Query to generate list of countries that don't use ppm
query1a = """SELECT distinct country, unit
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE unit != "ppm"
            ORDER BY country """
# Query to generate distinct list of pollutants with value of 0.0
query2 = """SELECT distinct pollutant, value
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE value = 0
            ORDER BY pollutant """

# Unit List Query
UnitList = Data_day1.query_to_pandas_safe(query1)
print(UnitList)

# Country-Unit List - no ppm
CountryUnitList = Data_day1.query_to_pandas_safe(query1a)
print(CountryUnitList)
# Pollutant List - 0 pollutant value only
PollutantList = Data_day1.query_to_pandas_safe(query2)
print(PollutantList)