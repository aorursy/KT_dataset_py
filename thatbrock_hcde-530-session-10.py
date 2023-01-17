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
# Set your own project id here

PROJECT_ID = 'hcde-530-270001'

from google.cloud import bigquery

client = bigquery.Client(project=PROJECT_ID)
# Construct a reference to the "chicago_crime" dataset

dataset_ref = client.dataset("hcde530_dataset", project=PROJECT_ID)



# API request - fetch the dataset

dataset = client.get_dataset(dataset_ref)
tables = list(client.list_tables(dataset))

for table in tables:  

    print(table.table_id)
table_ref = dataset_ref.table("original_cars")



# API request - fetch the table

table = client.get_table(table_ref)
table.schema
# Preview the first five lines of the "full" table

client.list_rows(table, max_results=5).to_dataframe()
df=client.list_rows(table, max_results=5).to_dataframe()

df.head(20)
# Write the query

query1 = """

          SELECT 

            DISTINCT Cylinders AS Cylinders, 

            Model AS Model,

            COUNT(Car) AS Count

          FROM 

            `hcde-530-270001.hcde530_dataset.original_cars` 

          GROUP BY 

            Cylinders, 

            Model

          ORDER BY 

            Cylinders, 

            Model DESC

        """
query_job1 = client.query(query1)



# Make an API request  to run the query and return a pandas DataFrame

housestyleAC = query_job1.to_dataframe()



# See the resulting table made from the query

print(housestyleAC)