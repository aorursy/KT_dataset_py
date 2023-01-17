# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

# from subprocess import check_output
# print(check_output(["ls", "../input"]))

# Any results you write to the current directory are saved as output.
import bq_helper
openaq = bq_helper.BigQueryHelper(active_project="bigquery-public-data", dataset_name='openaq')
openaq.list_tables()
openaq.table_schema('global_air_quality')
openaq.head('global_air_quality')
query = """SELECT DISTINCT country, unit
           FROM `bigquery-public-data.openaq.global_air_quality`
           WHERE LOWER(unit) != "ppm"
        """
print('Query size = {} MB'.format(openaq.estimate_query_size(query)*1000))
not_ppm = openaq.query_to_pandas_safe(query)
not_ppm
not_ppm.to_csv("countries_not_using_ppm.csv")
query2 = """SELECT DISTINCT pollutant, value
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE  value = 0
            """
print('Query size = {} MB'.format(openaq.estimate_query_size(query2)*1000))
no_pollutant = openaq.query_to_pandas_safe(query2)
no_pollutant
no_pollutant.to_csv("pollutants_with_zero_values.csv")
