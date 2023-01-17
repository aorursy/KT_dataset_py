# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import bq_helper  #BiqQuery helper package

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
# Any results you write to the current directory are saved as output.
open_aq = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="openaq")
open_aq.list_tables()
open_aq.head("global_air_quality")

query = """SELECT city
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE country = 'US'
            """

open_aq.estimate_query_size(query)
us_cities = open_aq.query_to_pandas_safe(query)
us_cities.head()
us_cities.city.value_counts().head()
#countries that *don't* use ppm as units
query = """SELECT country
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE unit != 'ppm'
            """
open_aq.estimate_query_size(query)
countries_not_ppm = open_aq.query_to_pandas_safe(query)
countries_not_ppm.head()
countries_not_ppm.country.unique()
#pollutants with a value of 0
query = """SELECT pollutant
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE value = 0
            """
open_aq.estimate_query_size(query)
pollutants_zero = open_aq.query_to_pandas_safe(query)
pollutants_zero.head()
pollutants_zero.pollutant.unique()
