# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import bq_helper
import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
traffic_data = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="nhtsa_traffic_fatalities")

query_a_2016 = "SELECT * FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2016`"
query_a_2015 = "SELECT * FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`"
traffic_data.estimate_query_size(query_a_2016)
traffic_data.estimate_query_size(query_a_2015)
accidents_2016_data = traffic_data.query_to_pandas_safe(query_a_2016)
accidents_2015_data = traffic_data.query_to_pandas_safe(query_a_2015)
accidents_2016_data.to_csv('accidents_2016.csv')
accidents_2015_data.to_csv('accidents_2015.csv')