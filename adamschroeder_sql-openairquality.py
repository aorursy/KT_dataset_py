# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
#print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
# import package with helper functions 
import bq_helper

# create a helper object for this dataset
open_aq = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="openaq")

# print all the tables in this dataset (there's only one!)
open_aq.list_tables()
open_aq.table_schema('global_air_quality')
open_aq.head("global_air_quality")
#I want to select all the values from the "city" column for the rows there the "country" column 
#is "us" (for "United States").

query = """ SELECT city
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE country = 'US'
        """
open_aq.estimate_query_size(query)
us_cities = open_aq.query_to_pandas_safe(query)
us_cities['city'].value_counts().head()
query2 = """ SELECT country
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE pollutant != 'ppm'
        """
open_aq.estimate_query_size(query2)
pollutant_not_ppm = open_aq.query_to_pandas(query2)
pollutant_not_ppm.tail()
pollutant_not_ppm['country'].unique()
# view the table
open_aq.head("global_air_quality")
query3 = """ SELECT pollutant, value
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE value = 0.00
        """
open_aq.estimate_query_size(query3)
print ('The 1st query to pandas failed due to size limit, but the 2nd worked.')
pltnt_value_0 = open_aq.query_to_pandas_safe(query3, max_gb_scanned=0.0001)
pltnt_value_0 = open_aq.query_to_pandas_safe(query3, max_gb_scanned=0.003)
print(pltnt_value_0['value'].value_counts())
pltnt_value_0['pollutant'].unique()
