# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import os
print(os.listdir("../input"))
import bq_helper
# Any results you write to the current directory are saved as output.
open_aq = bq_helper.BigQueryHelper(active_project="bigquery-public-data", dataset_name = "openaq")
open_aq.list_tables()
query = """SELECT city,value,location,pollutant,unit,timestamp
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE country = 'US' AND pollutant = 'co' AND timestamp > '2018-01-01'
            ORDER BY value DESC
        """
us_cities = open_aq.query_to_pandas_safe(query)
us_cities.head(200)
open_aq.head("global_air_quality", num_rows=5)