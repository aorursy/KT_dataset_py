# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory


# Any results you write to the current directory are saved as output.
from google.cloud import bigquery
from bq_helper import BigQueryHelper


bq_openaq = BigQueryHelper('bigquery-public-data', 'openaq')
# print a list of all the tables in the hacker_news dataset
bq_openaq.list_tables()
# show schema of table
bq_openaq.table_schema("global_air_quality")

bq_openaq.head('global_air_quality')



query =  ''' 
             SELECT distinct country
             FROM `bigquery-public-data.openaq.global_air_quality`
             WHERE unit != 'ppm' 

        '''

countries_pollution = bq_openaq.query_to_pandas(query)
countries_pollution


query_01 =  ''' 
             SELECT distinct country
             FROM `bigquery-public-data.openaq.global_air_quality`
             WHERE value = 0 

        '''

zero_pollutants = bq_openaq.query_to_pandas(query_01) 
zero_pollutants 
