# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import bq_helper
open_aq= bq_helper.BigQueryHelper(active_project="bigquery-public-data",dataset_name="openaq") 
open_aq.list_tables()
open_aq.table_schema("global_air_quality")
open_aq.head("global_air_quality")


query = """ SELECT country from `bigquery-public-data.openaq.global_air_quality` where unit !='ppm' """
open_aq.estimate_query_size(query)
open_aq.query_to_pandas_safe(query)
query1= """SELECT pollutant from `bigquery-public-data.openaq.global_air_quality` where value=0 """
open_aq.query_to_pandas_safe(query1)