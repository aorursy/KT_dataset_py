# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import bq_helper
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

openaq = bq_helper.BigQueryHelper(active_project= "bigquery-public-data", 
                                       dataset_name = "openaq")

#openaq.head("global_air_quality", selected_columns="unit", num_rows=100)
openaq.head("global_air_quality")


query1 = "select country, unit from `bigquery-public-data.openaq.global_air_quality` where unit != 'ppm'"

no_ppm = openaq.query_to_pandas_safe(query1)

print(no_ppm)
no_ppm.country.value_counts().head()
query2 = "select country, pollutant from `bigquery-public-data.openaq.global_air_quality` where value = 0"
zero_poll = openaq.query_to_pandas_safe(query2)

print(zero_poll)
zero_poll.country.value_counts().head()