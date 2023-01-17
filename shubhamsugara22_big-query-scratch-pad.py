from google.cloud import bigquery
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
#create "client " object

client=bigquery.Client()
# construct a reference to get dataset

dataset_ref=client.dataset("hacker_news",project="bigquery-public-data")

#api request fetvh the dataset

dataset=client.get_dataset(dataset_ref)
#list of all the tables in hacker news

tables=list(client.list_tables(dataset))

#print the name of all table 

for table in tables:

    print(table.table_id) 
#construct a refernce for the table

table_ref=dataset_ref.table("full")

# api request to fetch thetable

table=client.get_table(table_ref)

table.schema
client.list_rows(table,max_results=5).to_dataframe()
