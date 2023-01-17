

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import bq_helper

import os
print(os.listdir("../input"))


#Create a helper object 
open_aq = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="openaq")

#Lets see the tables in the dataset
open_aq.list_tables()
open_aq.head("global_air_quality")
query = """SELECT city
           FROM `bigquery-public-data.openaq.global_air_quality`
           WHERE country = 'US'
        """
us_cities = open_aq.query_to_pandas(query)
us_cities.city.value_counts().head()
query_pol = """SELECT country
               FROM `bigquery-public-data.openaq.global_air_quality`
               WHERE unit != 'ppm'
             
            """
non_ppm_nations = open_aq.query_to_pandas(query_pol)
non_ppm_nations['country'].value_counts()
query_0 = """SELECT country
               FROM `bigquery-public-data.openaq.global_air_quality`
               WHERE value = 0.0
             
            """
non_pol = open_aq.query_to_pandas(query_0)
non_pol['country'].value_counts()