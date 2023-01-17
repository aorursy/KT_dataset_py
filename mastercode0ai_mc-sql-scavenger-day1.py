# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import bq_helper

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

#from subprocess import check_output
#print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
open_aq = bq_helper.BigQueryHelper(active_project="bigquery-public-data", dataset_name="openaq")
open_aq.list_tables()
open_aq.head("global_air_quality")
query1 = """SELECT country
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE pollutant != "ppm" """
#open_aq.estimate_query_size(query1)
no_ppm_pollutants = open_aq.query_to_pandas_safe(query1)
no_ppm_countries = no_ppm_pollutants.country.unique()
print("There are " + str(len(no_ppm_countries)) + " countries that use pollutant other than ppm")
print("Below is the list of that countries:")
no_ppm_countries
query2 = """SELECT pollutant
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE value = 0"""
zero_pollutants = open_aq.query_to_pandas_safe(query2)
print("There are " + str(len(zero_pollutants.pollutant.unique())) + " pollutants that have value of 0 (zero)")
print("Below is the list of these pollutants")
zero_pollutants.pollutant.unique()
