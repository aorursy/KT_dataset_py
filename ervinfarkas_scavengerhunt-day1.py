# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

#from subprocess import check_output
#print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
# import our bq_helper package
import bq_helper 
# create a helper object for our bigquery dataset
openaq = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="openaq")

# print a list of all the tables in the OpenAQ dataset
openaq.list_tables()
# print information on all the columns in the 'global_air_quality' table
# in the hacker_news dataset
openaq.table_schema('global_air_quality')
# preview the first couple lines of the 'global_air_quality' table
openaq.head('global_air_quality')
# preview the first 50 entries in the 'unit' column of the 'global_air_quality' table
openaq.head('global_air_quality', selected_columns='unit', num_rows=50)
# this query looks in the 'global_air_quality' table in the openaq
# dataset, then gets which countries use a unit other than ppm to measure any type of pollution
query = """SELECT country
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE unit != 'ppm' 
        """

# check how big this query will be
openaq.estimate_query_size(query)
#this query looks in the 'global_air_quality' table in the openaq
# dataset, then gets which countries use a unit other than ppm to measure any type of pollution
#make DISTINCT computation in SQL
query_dist = """SELECT DISTINCT country, unit
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE unit != 'ppm' 
        """

# check how big this query will be
openaq.estimate_query_size(query_dist)
# only run this query if it's less than 100 MB and assign to countries dataframe 
countries = openaq.query_to_pandas_safe(query_dist, max_gb_scanned=0.1)
print('List of countries that use a unit other than ppm to measure any type of pollution:')
print(countries.sort_values('country').reset_index()[['country', 'unit']])


#save to csv
countries.sort_values('country').reset_index()[['country', 'unit']].to_csv("day1.csv")
#Which pollutants have a value of exactly 0
query_dist = """SELECT DISTINCT pollutant, value
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE value = 0 
        """

# check how big this query will be
openaq.estimate_query_size(query_dist)
# only run this query if it's less than 100 MB and assign to pollutants dataframe 
pollutants = openaq.query_to_pandas_safe(query_dist, max_gb_scanned=0.1)
print (100*"-"+"\n")
print('List of pollutants that have a value of 0:')
print(pollutants.sort_values('pollutant').reset_index()[['pollutant', 'value']])
# add to csv 
line1= 100*"-"+"\n"
line2='List of pollutants that have a value of 0:'
with open('day1.csv', 'a') as f:
    f.write(line1)
    f.write(line2)
    pollutants.sort_values('pollutant').reset_index()[['pollutant', 'value']].to_csv(f)
