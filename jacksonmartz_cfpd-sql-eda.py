# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt


# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
from google.cloud import bigquery
client = bigquery.Client()

data_ref = client.dataset("cfpb_complaints", project="bigquery-public-data")

# API request
data = client.get_dataset(data_ref)

table_ref = data_ref.table("complaint_database")

# API request
table = client.get_table(table_ref)

# Preview the first five lines of the table
df = client.list_rows(table, max_results=10000).to_dataframe()
df.head(5)
# What is the ratio of timely and untimely responses to complaints?

query = '''
        SELECT timely_response, COUNT(*) AS count
        FROM `bigquery-public-data.cfpb_complaints.complaint_database`
        GROUP BY timely_response
        '''

res = client.query(query).result().to_dataframe()
res
# Ratio of disputes coming in for a timely response vs an untimely response.

q2 = '''
     SELECT 
         cd.timely_response,
         t1.count AS total_count,
         COUNT(consumer_disputed) AS disputed_count,
         (COUNT(consumer_disputed) / t1.count) AS disputed_ratio
     FROM `bigquery-public-data.cfpb_complaints.complaint_database` as cd
     JOIN
         (SELECT timely_response, COUNT(*) AS count
         FROM `bigquery-public-data.cfpb_complaints.complaint_database`
         GROUP BY timely_response) as t1
     ON t1.timely_response = cd.timely_response
     WHERE consumer_disputed = True
     GROUP BY timely_response, total_count
     '''
r2 = client.query(q2).result().to_dataframe()
r2
# Which subproducts result in the largest number of complaints?
# Determine the product types that are most frequently reported in the CFPB complaint database.

query_1 = '''
        SELECT
          product,
          subproduct,
          COUNT(DISTINCT(complaint_id)) AS count_complaints
        FROM
          `bigquery-public-data.cfpb_complaints.complaint_database` 
        GROUP BY
          product, subproduct
        ORDER BY
          count_complaints desc
          '''

res_1 = client.query(query_1).result().to_dataframe()

res_1.head()
def run_query(query):
    '''Automating the repetitive process of turning queries into workable pandas dataframes'''
    result = client.query(query).result().to_dataframe()
    return result
# States with most complaints in 2019

state_query = '''
              SELECT state, COUNT(*) AS count
              FROM `bigquery-public-data.cfpb_complaints.complaint_database` 
              WHERE date_received BETWEEN '2019-01-01' AND '2020-01-01'
              GROUP BY state
              ORDER BY count DESC
              LIMIT 10
              '''

# using simple functioned defined in cell above to reduce code
run_query(state_query)
top_5_states = list(run_query(state_query)['state'].head(5))
top_5_states
# Find top product issues in the top 5 states

df.columns = df.columns.str.strip()

top_states_df = df[df['state'].isin(top_5_states)]

grouped = top_states_df.groupby(['state', 'product'])['product'].agg('count')
states_prods = pd.DataFrame(grouped)
states_prods.columns = ['count']
states_prods
# Visualizing count of each states product complaints

for x in top_5_states:
    states_prods.loc[x].plot(kind='bar')
