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

dataset_ref = client.dataset('hacker_news', project='bigquery-public-data')

dataset = client.get_dataset(dataset_ref)

tables = list(client.list_tables(dataset))



# iterate through the tables and print the table id, for eg.

for table in tables:

    print(table.table_id)
comments_table_ref = dataset_ref.table('comments')

comments_table = client.get_table(comments_table_ref)

comments_table.schema
client.list_rows(comments_table, max_results=5).to_dataframe()
popular_query = """

                SELECT parent, COUNT(id) AS comment_count

                FROM `bigquery-public-data.hacker_news.comments`

                GROUP BY parent

                HAVING COUNT(id) > 10

"""

popular_query_res = client.query(popular_query).to_dataframe()
popular_query_res.head()