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
from google.cloud import bigquery



client = bigquery.Client()
# Reference

dataset_ref = client.dataset("nhtsa_traffic_fatalities",

                             project = "bigquery-public-data")



# API

dataset = client.get_dataset(dataset_ref)



# List

tables = list(client.list_tables(dataset))



for table in tables:

    print(table.table_id)
table1_ref = dataset_ref.table("accident_2016")



table1 = client.get_table(table1_ref)



client.list_rows(table1,

                max_results = 5).to_dataframe()
# Query - accidents per day 2016



query = """

        SELECT COUNT(consecutive_number) AS num_accidents,

        EXTRACT(DAYOFWEEK FROM timestamp_of_crash) AS day_of_week

        FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2016`

        GROUP BY day_of_week

        ORDER BY num_accidents DESC

        """

# Safe config 



safe_config = bigquery.QueryJobConfig(maximum_bytes_billed = 10**10)



query_job = client.query(query,

                        job_config = safe_config)



query_results = query_job.to_dataframe()



query_results
# Table-2015



table2_ref = dataset_ref.table("accident_2015")



table2 = client.get_table(table2_ref)



client.list_rows(table2,

                max_results = 5).to_dataframe()
# Query - 2015 



query1 = """

         SELECT COUNT(consecutive_number) AS num_accidents,

         EXTRACT(DAYOFWEEK FROM timestamp_of_crash) AS day_of_week

         FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`

         GROUP BY day_of_week

         ORDER BY num_accidents DESC

         """



safe_config = bigquery.QueryJobConfig(maximum_bytes_billed = 10**10)



query1_job = client.query(query1,

                         job_config = safe_config)



query1_results = query1_job.to_dataframe()



query1_results
# 2016 MONTH



query3 = """

            SELECT COUNT(consecutive_number) AS num_accidents,

            EXTRACT(MONTH FROM timestamp_of_crash) AS month

            FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2016`

            GROUP BY month

            ORDER BY num_accidents DESC 

            """



safe_config = bigquery.QueryJobConfig(maximum_bytes_billed = 10**10)



query3_job = client.query(query3,

                         job_config = safe_config)



query3_results = query3_job.to_dataframe()



query3_results
# Month - 2015



query4 = """

            SELECT COUNT(consecutive_number) AS num_accidents,

            EXTRACT(MONTH FROM timestamp_of_crash) AS month

            FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`

            GROUP BY month

            ORDER BY num_accidents DESC

            """



safe_config = bigquery.QueryJobConfig(maximum_bytes_billed = 10**10)



query4_job = client.query(query4,

                         job_config = safe_config)



query4_results = query4_job.to_dataframe()



query4_results