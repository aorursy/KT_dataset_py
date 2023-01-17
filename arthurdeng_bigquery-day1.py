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
import bq_helper
OpenAQ = bq_helper.BigQueryHelper('bigquery-public-data','openaq')
OpenAQ.list_tables()
OpenAQ.head('global_air_quality')
query = """SELECT city
FROM `bigquery-public-data.openaq.global_air_quality` 
WHERE country = 'US' 
"""
us_cities = OpenAQ.query_to_pandas_safe(query)
us_cities.head()

query_ppm = """SELECT country
FROM `bigquery-public-data.openaq.global_air_quality` 
WHERE unit != 'ppm' 
"""
ppm = OpenAQ.query_to_pandas_safe(query_ppm)
ppm.drop_duplicates()
query_pollutant = """SELECT pollutant
FROM `bigquery-public-data.openaq.global_air_quality` 
WHERE value = 0.00 
"""
pollutant = OpenAQ.query_to_pandas_safe(query_pollutant)
pollutant.drop_duplicates()
