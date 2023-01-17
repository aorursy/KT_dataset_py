# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import bq_helper
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
open_aq = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="openaq")

# print all the tables in this dataset (there's only one!)
open_aq.list_tables()

# Hint: to get rows where the value *isn't* something, use "!="
query1 = """SELECT country
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE unit != 'ppm'
        """
# check how big this query will be
open_aq.estimate_query_size(query1)

countries_notppm = open_aq.query_to_pandas_safe(query1)

countries_notppm.country.unique()

countries_notppm.country.nunique()

query2 = """SELECT country
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE unit = 'ppm'
        """
open_aq.estimate_query_size(query1)

countries_notppm = open_aq.query_to_pandas_safe(query1)

countries_notppm.country.unique()


#from subprocess import check_output
#print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


