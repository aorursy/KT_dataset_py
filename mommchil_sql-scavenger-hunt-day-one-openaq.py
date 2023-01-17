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
OpenAQ = bq_helper.BigQueryHelper(active_project= "bigquery-public-data", 
                                       dataset_name = "openaq")
# list tables
OpenAQ.list_tables()
# check out the first lines from the global_air_quality table
OpenAQ.head("global_air_quality")
# look at the global_air_quality table in the OpenAQ dataset,
# then get the country column from every row where 
# the unit column is not equal to ppm.
query = """SELECT country
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE unit != "ppm" """
# save the query as a Pandas dataframe
countries = OpenAQ.query_to_pandas_safe(query)

# print the unique list of countries that match the criteria
countries.country.unique()
# look at the global_air_quality table in the OpenAQ dataset,
# then get the pollutant column from every row where 
# the value column is exactly equal to '0'.
query_pollutant = """SELECT pollutant
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE value = 0 """
# save the query as a Pandas dataframe
pollutants = OpenAQ.query_to_pandas_safe(query_pollutant)

# print the unique list of pollutant that match the criteria
pollutants.pollutant.unique()