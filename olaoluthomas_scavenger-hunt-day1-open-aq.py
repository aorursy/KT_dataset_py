# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
# print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
import bq_helper

# helper object
open_aq = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                  dataset_name="openaq")

open_aq.list_tables()
open_aq.head("global_air_quality")
query_1 = """SELECT city
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE country = 'US'
        """

us_cities = open_aq.query_to_pandas_safe(query_1)
us_cities.city.value_counts().head()
query_2 = """SELECT country, pollutant
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE pollutant NOT LIKE '%pm%'
            """

non_pm_countries = open_aq.query_to_pandas_safe(query_2)
non_pm_countries
non_pm_countries.country.value_counts()
query_3 = """SELECT pollutant, value
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE value = 0
            """

zero_value = open_aq.query_to_pandas_safe(query_3)
zero_value.pollutant.value_counts()
