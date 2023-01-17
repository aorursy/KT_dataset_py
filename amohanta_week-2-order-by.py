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



#Create a 'Client' Object

client = bigquery.Client()
# Construct a reference to the "US Traffic Fatality Records database"

dataset_ref = client.dataset("nhtsa_traffic_fatalities", project = 'bigquery-public-data')



#API request - fetch the dataset

dataset = client.get_dataset(dataset_ref)
# List all the tables in the "nhtsa_traffic_fatalities" dataset

tables = list(client.list_tables(dataset))

print(type(tables))



#print names of all tables in the dataset 

for table in tables:

    print(table.table_id)
# Construct a reference to the "accident_2015" table

table_ref = dataset_ref.table("accident_2015")



# API request - fetch the table

table = client.get_table(table_ref)



#print the information on all the columns in the "accident_2015" table in the nhtsa_traffic_fatalities dataset

table.schema



client.list_rows(table).to_dataframe().shape
import pandas as pd

accident_2015 = client.list_rows(table).to_dataframe()



#save a file

with open ("acc_2015.csv", "w") as file:

    file.write(accident_2015.to_csv())
# Preview the first five lines of the "accident_2015" table

client.list_rows(table, max_results=5).to_dataframe()
# Query to find out the number of accidents per each day of the week

query = """

        SELECT COUNT(consecutive_number) AS num_accidents, 

               EXTRACT(DAYOFWEEK FROM timestamp_of_crash) AS day_of_week

        FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`

        GROUP BY day_of_week

        ORDER BY num_accidents DESC

        """
# Set up the query (cancel the query if it would use too much of 

# your quota, with the limit set to 1 GB)

safe_config = bigquery.QueryJobConfig(maximum_bytes_billed=10**9)

query_job = client.query(query, job_config=safe_config)



# API request - run the query, and convert the results to a pandas DataFrame

accidents_by_day = query_job.to_dataframe()



# Print the DataFrame

accidents_by_day
accident_2015.head()
df = accident_2015[['consecutive_number', 'timestamp_of_crash' ]]

df.head()
df['day_of_week'] = df['timestamp_of_crash'].dt.weekday_name

df.head()
from collections import Counter

dict_day=Counter(df.iloc[:,2])
# Arrange dictionary in desecending order of values

dict_sort=sorted(dict_day.items(), key = lambda x: x[1], reverse=True)

dict_sort
acc_by_day = pd.DataFrame(dict_sort)

acc_by_day.columns = ['day_of_week','num_accidents',]

acc_by_day = acc_by_day[['num_accidents','day_of_week']]

acc_by_day
accident_by_day = pd.DataFrame (df.iloc[:,2].value_counts())

accident_by_day.reset_index(inplace = True)

accident_by_day.columns = ['day_of_week','num_accidents']

accident_by_day = accident_by_day[['num_accidents','day_of_week']]

accident_by_day