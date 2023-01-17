# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
from google.cloud import bigquery



# create client object

client = bigquery.Client()



# create a handle to the dataset

dataset_ref = client.dataset('nhtsa_traffic_fatalities', project='bigquery-public-data')



# fetch the dataset

dataset = client.get_dataset(dataset_ref)



# list the interesting info from the dataset (say the tables in the dataset)

tables = list(client.list_tables(dataset))

print ("number of tables: ", len(tables))



for table in tables:

    print (table.table_id)
# we shall analyze accident_2015 table

accident_2015_table_ref = dataset_ref.table('accident_2015')



# fetch the table

accident_2015_table = client.get_table(accident_2015_table_ref)



# get the structure of the table

# accident_2015_table.schema



# view the first few rows of the table

client.list_rows(accident_2015_table, max_results=5).to_dataframe()
# let us start writing queries. let us write a query to find out the number of accidents for each day of the week

each_day_query = """

        SELECT COUNT(consecutive_number) AS num_accidents, EXTRACT (DAYOFWEEK from timestamp_of_crash) AS day_of_week

        FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`

        GROUP BY day_of_week

        ORDER BY num_accidents DESC

"""





# safe config settings 

safe_config = bigquery.QueryJobConfig(maximum_bytes_billed=10**9)



# configure the query

each_day_query_job = client.query(each_day_query, job_config=safe_config)



import pandas as pd

# execute the query

accidents_by_day = each_day_query_job.to_dataframe()



# get the results

accidents_by_day