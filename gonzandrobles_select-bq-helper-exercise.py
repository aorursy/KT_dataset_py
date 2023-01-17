# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import bq_helper

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
#print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
open_aq = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="openaq")
open_aq.list_tables()
open_aq.head("global_air_quality")
query = """SELECT city
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE country = 'US'
        """
us_cities = open_aq.query_to_pandas_safe(query)
us_cities.city.value_counts().head()
query_2 = '''SELECT country
             FROM `bigquery-public-data.openaq.global_air_quality`
             WHERE pollutant != "ppm"'''
country_poll = open_aq.query_to_pandas_safe(query_2)
country_poll['country'].value_counts().head(10)
zero = """
SELECT DISTINCT pollutant, value
FROM `bigquery-public-data.openaq.global_air_quality`
WHERE value = 0
"""
zero_pol_readings = open_aq.query_to_pandas_safe(zero)
zero_pol_readings
