### This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import bq_helper 

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

#from subprocess import check_output
#print(check_output(["ls", "../input"]).decode("utf8"))
open_aq = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="openaq")
query = """SELECT city
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE country = 'US'
        """
us_cities = open_aq.query_to_pandas_safe(query)
us_cities.city.value_counts().head()
query_ppm = """SELECT country
                FROM `bigquery-public-data.openaq.global_air_quality`
                WHERE unit != 'ppm'
                """
us_countries_ppm = open_aq.query_to_pandas_safe(query_ppm)
us_countries_ppm.country.value_counts().head()

query_pollutant = """ SELECT pollutant 
                       FROM `bigquery-public-data.openaq.global_air_quality`
                       WHERE value = 0
                       """
zero_pollutants = open_aq.query_to_pandas_safe(query_pollutant)
zero_pollutants.pollutant.unique()
# Any results you write to the current directory are saved as output.