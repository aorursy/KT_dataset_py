# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import bq_helper

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
scav1 = """SELECT country
           FROM `bigquery-public-data.openaq.global_air_quality`
           WHERE unit != 'ppm'
        """

uniq_nat = open_aq.query_to_pandas_safe(scav1)
list(uniq_nat.country.value_counts().axes[0])
scav2 = """SELECT pollutant
           FROM `bigquery-public-data.openaq.global_air_quality`
           WHERE value = 0.00
        """

null_poll = open_aq.query_to_pandas_safe(scav2)
list(null_poll["pollutant"].unique())