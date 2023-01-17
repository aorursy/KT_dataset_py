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



# Create a "Client" object

client = bigquery.Client()



# Construct a reference to the "ncaa_basketball" dataset

dataset_ref = client.dataset("ncaa_basketball", project="bigquery-public-data")



# API request - fetch the dataset

dataset = client.get_dataset(dataset_ref)
# Get a list of the tables in the dataset



tables = list(client.list_tables(dataset))

for table in tables:

    print(table.table_id)

# Oooooooo...mascots. Fun! Let's take a look in there.

# Fetch the mascots

table_ref = dataset_ref.table("mascots")

table = client.get_table(table_ref)



# Print the first five rows

client.list_rows(table, max_results = 5).to_dataframe()
# Kaboom! is my favorite so far. Let's see what types of mascots are most popular. 



query = """ SELECT mascot, COUNT(mascot) total

            FROM `bigquery-public-data.ncaa_basketball.mascots`

            GROUP BY mascot

            ORDER BY COUNT(mascot) DESC



"""

client = bigquery.Client()

query_job = client.query(query)

mascot_frame = query_job.to_dataframe()

mascot_frame.head()
# Bulldogs, Tigers, and Eagles. (Wizard of Oz here.)



# Let's see if any of the mascot names are repeated in the database. Using the number method here instead of the full name in the COUNT().

query = """

            SELECT mascot_name, COUNT(1) count 

            FROM `bigquery-public-data.ncaa_basketball.mascots`

            WHERE NOT (mascot_name = "None")

            GROUP BY 1

            HAVING COUNT(1) > 1

            ORDER BY 2 DESC

"""



client = bigquery.Client()

query_job = client.query(query)

mascot_frame = query_job.to_dataframe()

mascot_frame.head()



# BTW, the None's are the most popular. But Bucky, pride of Wisconsin!, right up there. 

# The 'non_tax_type' one is calling me. I'll get a full list of those. 
query = """

            SELECT non_tax_type, COUNT(1) count 

            FROM `bigquery-public-data.ncaa_basketball.mascots`

            GROUP BY 1

            ORDER BY 2 DESC

"""



client = bigquery.Client()

query_job = client.query(query)

query_job.to_dataframe()

# Removed the mascot_frame as I wasn't going for just the head anymore.
# And because I gotta.....



query = """

            SELECT mascot_name, market, name 

            FROM `bigquery-public-data.ncaa_basketball.mascots`

            WHERE non_tax_type = "Clergy" OR non_tax_type = "Entrepreneurs"

"""



client = bigquery.Client()

query_job = client.query(query)

query_job.to_dataframe()
