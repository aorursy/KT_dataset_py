import bq_helper 
openaq = bq_helper.BigQueryHelper(active_project= "bigquery-public-data", 
                                       dataset_name = "openaq")

openaq.list_tables()
openaq.table_schema('global_air_quality')
openaq.head('global_air_quality')

#query 1
query = """SELECT country
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE unit != "ppm" 
            group by country"""

#query 2
query = """SELECT pollutant
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE value = 0 
            group by pollutant"""


openaq.estimate_query_size(query)

openaq.query_to_pandas_safe(query)
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.