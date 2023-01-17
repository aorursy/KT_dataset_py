from google.cloud import bigquery

import numpy as np

import pandas as pd
client = bigquery.Client()

dataset_ref = client.dataset("chicago_taxi_trips", project="bigquery-public-data")

taxi_dat = client.get_dataset(dataset_ref)
tables = list(client.list_tables(taxi_dat))

for table in tables:

    print(table.table_id)
table_ref = dataset_ref.table('taxi_trips')

table = client.get_table(table_ref)

table.schema
## First to see how many payment methods there are total

preliminary_query ="""

SELECT DISTINCT payment_type

FROM `bigquery-public-data.chicago_taxi_trips.taxi_trips`

""" 



safe_config = bigquery.QueryJobConfig(maximum_bytes_billed=10**10)

preliminary_query_job =client.query(preliminary_query,job_config=safe_config)

preliminary_query_result = preliminary_query_job.to_dataframe()

preliminary_query_result.head(10)

query ="""

SELECT payment_type, COUNT(1) AS transactions

FROM `bigquery-public-data.chicago_taxi_trips.taxi_trips`

GROUP BY payment_type

ORDER BY transactions DESC

""" 



safe_config = bigquery.QueryJobConfig(maximum_bytes_billed=10**10)

query_job =client.query(query,job_config=safe_config)

query_result = query_job.to_dataframe()

query_result.head(10)
query ="""

SELECT payment_type, COUNT(1) AS transactions

FROM `bigquery-public-data.chicago_taxi_trips.taxi_trips`

WHERE payment_type = 'Credit Card' OR payment_type = 'Cash'

GROUP BY payment_type

ORDER BY transactions DESC

""" 



safe_config = bigquery.QueryJobConfig(maximum_bytes_billed=10**10)

query_job =client.query(query,job_config=safe_config)

query_result = query_job.to_dataframe()

query_result.head(10)
import matplotlib.pyplot as plt

import seaborn as sns

plt.subplots(figsize=(12, 2))

sns.barplot(x=query_result['transactions'],y=query_result['payment_type'], palette='Blues_r')

#generating the graph pictured above
query ="""

SELECT EXTRACT(YEAR FROM trip_start_timestamp) AS year, payment_type, COUNT(1) AS transactions

FROM `bigquery-public-data.chicago_taxi_trips.taxi_trips`

WHERE payment_type = 'Credit Card' OR payment_type = 'Cash'

GROUP BY year, payment_type

ORDER BY year, transactions DESC

""" 



safe_config = bigquery.QueryJobConfig(maximum_bytes_billed=10**10)

query_job =client.query(query,job_config=safe_config)

query_result = query_job.to_dataframe()

query_result.head(10)
years = [2013,2014,2015,2016,2017,2018,2019,2020]

ind = 0

cash = []

cc = []

for num in query_result.transactions:

    if ind % 2 == 0:

        cash = cash + [num]

        ind += 1

    else:

        cc = cc + [num]

        ind += 1

cash = pd.Series(cash, index=years, name='Cash')

cash = pd.DataFrame(cash, index=cash.index)

cc = pd.Series(cc, index=years, name='Credit_Card')

cc = pd.DataFrame(cc, index=cash.index)



#The irony of using python for a join when I initially chose this project to practice SQL is not lost on me

pay_df = pd.merge(cash, cc, left_index=True, right_index=True)

pay_df
plt.subplots(figsize=(10, 8))

sns.lineplot(data=pay_df, palette="Blues_r", linewidth=3.5)
