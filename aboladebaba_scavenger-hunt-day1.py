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
global_air_quality = bq_helper.BigQueryHelper(active_project="bigquery-public-data", 
                                              dataset_name='openaq')
global_air_quality.list_tables()
global_air_quality.table_schema('global_air_quality')
# what are the different units being used by countries
QUERY = """SELECT DISTINCT unit FROM `bigquery-public-data.openaq.global_air_quality`"""
global_air_quality.estimate_query_size(query=QUERY)
global_air_quality.query_to_pandas(query=QUERY)
# the distinct here will eliminate repeating country codes
QUERY = """SELECT DISTINCT country, unit 
FROM `bigquery-public-data.openaq.global_air_quality`
WHERE unit != 'ppm'"""
# check the scan estimate
global_air_quality.estimate_query_size(query=QUERY)
global_air_quality.query_to_pandas(query=QUERY)
QUERY = """SELECT DISTINCT pollutant, value 
FROM `bigquery-public-data.openaq.global_air_quality`
WHERE value = 0 """
global_air_quality.estimate_query_size(query=QUERY)
global_air_quality.query_to_pandas(QUERY)
