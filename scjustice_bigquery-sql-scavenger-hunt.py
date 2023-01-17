# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



# import our bq_helper package

import bq_helper



import os

#print(os.listdir("../input"))
## Day 1 Select, From, and Where

open_aq = bq_helper.BigQueryHelper(active_project="bigquery-public-data", 

                                   dataset_name="openaq")

open_aq.list_tables()

open_aq.head('global_air_quality')
# query to select all the items from the "city" column where the

# "country" column is "us"







query = '''SELECT city 

            FROM `bigquery-public-data.openaq.global_air_quality`

            WHERE country = "US"'''



print(open_aq.estimate_query_size(query))



us_cities = open_aq.query_to_pandas_safe(query)



us_cities.city.shape
query = """SELECT country, unit

            FROM `bigquery-public-data.openaq.global_air_quality`

            WHERE unit != 'ppm'"""





print(open_aq.estimate_query_size(query))

non_ppm_cities = open_aq.query_to_pandas_safe(query)
non_ppm_cities.shape
query = """SELECT DISTINCT pollutant

            FROM `bigquery-public-data.openaq.global_air_quality`

            WHERE value = 0"""



print(open_aq.estimate_query_size(query))

pollutant_query = open_aq.query_to_pandas_safe(query)
pollutant_query.head()
# Get the city and country for the measurements where the air qualit

# is the worst for each pollutant

query = """SELECT city, country, pollutant, value

            FROM `bigquery-public-data.openaq.global_air_quality`

            WHERE (pollutant, value) 

                IN (SELECT (pollutant, MAX(value)) AS max_value

                    FROM `bigquery-public-data.openaq.global_air_quality`

                    GROUP BY pollutant)"""

print(open_aq.estimate_query_size(query))

max_query = open_aq.query_to_pandas_safe(query)
max_query.head()
# Find the pollutant with sum less than 1000

query = """SELECT pollutant, SUM(value) AS sum

            FROM `bigquery-public-data.openaq.global_air_quality`

            GROUP BY pollutant

            HAVING sum < 1000"""



print(open_aq.estimate_query_size(query))

sum_query = open_aq.query_to_pandas_safe(query)
sum_query.head(10)
# Day 2 Group BY, Having, Count

# Any results you write to the current directory are saved as output.



hacker_news = bq_helper.BigQueryHelper(active_project= 'bigquery-public-data', 

                                       dataset_name= 'hacker_news')



hacker_news.list_tables()
hacker_news.head('comments')
query = """SELECT parent, COUNT(id) AS count

            FROM `bigquery-public-data.hacker_news.comments`

            GROUP BY parent

            HAVING COUNT(id) > 10"""



print(hacker_news.estimate_query_size(query))

popular_stories = hacker_news.query_to_pandas_safe(query)
popular_stories.head()
hacker_news.head('full')
query = """SELECT type, COUNT(id) AS count

            FROM `bigquery-public-data.hacker_news.full`

            GROUP BY type"""



hacker_news.estimate_query_size(query)

group_by_type = hacker_news.query_to_pandas_safe(query)
group_by_type
query = """SELECT deleted, COUNT(id) AS count

            FROM `bigquery-public-data.hacker_news.comments`

            WHERE deleted = True

            GROUP BY deleted

            """



print(hacker_news.estimate_query_size(query))

deleted_count = hacker_news.query_to_pandas_safe(query)
deleted_count
# Day 3

accidents = bq_helper.BigQueryHelper(active_project="bigquery-public-data", 

                                     dataset_name="nhtsa_traffic_fatalities")
accidents.list_tables()
accidents.head('accident_2015')
query = """SELECT COUNT(consecutive_number) AS accident_count, 

                  EXTRACT(DAYOFWEEK FROM timestamp_of_crash) AS day_of_week

            FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`

            GROUP BY day_of_week

            ORDER BY accident_count DESC"""

print(accidents.estimate_query_size(query))

day_count = accidents.query_to_pandas_safe(query)
day_count
import matplotlib.pyplot as plt

#plt.plot(data=day_count, x='day_of_week', y='accident_count')

plt.plot(day_count.accident_count)

plt.title("Number of Accidents by Rank of Day\n(Most to Least Dangerous)")
query = """SELECT COUNT(consecutive_number) as accident_count,

                  EXTRACT(HOUR FROM timestamp_of_crash) AS hour

            FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`

            GROUP BY hour

            ORDER BY accident_count DESC"""



accidents.estimate_query_size(query)

hour_count = accidents.query_to_pandas_safe(query)
plt.scatter(data=hour_count, y='accident_count', x='hour')
accidents.head('vehicle_2015')
query = """SELECT registration_state_name, COUNT(consecutive_number) as accident_count

            FROM `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_2015`

            WHERE hit_and_run = 'Yes'

            GROUP BY registration_state_name

            ORDER BY accident_count DESC"""



accidents.estimate_query_size(query)

hit_and_run = accidents.query_to_pandas_safe(query)
hit_and_run.head()
# Day 4

bitcoin_blockchain = bq_helper.BigQueryHelper(active_project=

                                              "bigquery-public-data",

                                              dataset_name=

                                              "crypto_bitcoin")
bitcoin_blockchain.list_tables()

bitcoin_blockchain.head('transactions')
query = """ WITH time AS

            (

                SELECT block_timestamp, "hash" AS transaction_id

                FROM `bigquery-public-data.crypto_bitcoin.transactions`

            )

            SELECT COUNT(transaction_id) AS transactions,

                EXTRACT(month FROM block_timestamp) AS month,

                EXTRACT(year FROM block_timestamp) AS year

            FROM time

            GROUP BY year, month

            ORDER BY year, month

        """



print(bitcoin_blockchain.estimate_query_size(query))

transactions_per_month = bitcoin_blockchain.query_to_pandas_safe(query=query, 

                                                                 max_gb_scanned=5)
transactions_per_month.head()
import matplotlib.pyplot as plt



plt.plot(transactions_per_month.transactions)

plt.title('Monthly Bitcoin Transactions')
query = """ WITH time AS

            (

                SELECT EXTRACT(dayofyear FROM block_timestamp) AS day,

                    EXTRACT(year FROM block_timestamp) AS year,

                    'hash' AS transaction_id

                FROM `bigquery-public-data.crypto_bitcoin.transactions`

            )

            SELECT COUNT(transaction_id) AS transactions, 

                day, year

            FROM time 

            WHERE year = 2017

            GROUP BY year, day

            ORDER BY year, day

        """



bitcoin_blockchain.estimate_query_size(query)
transactions_by_day = bitcoin_blockchain.query_to_pandas_safe(query, max_gb_scanned=5)
transactions_by_day.head()
import matplotlib.pyplot as plt



plt.plot(transactions_by_day.transactions)

plt.title('Bitcoin Transaction Count by Day During 2017')
# Try to get the number of transactions per merkle root using the blocks table



query = """ WITH blocks AS

            (

                SELECT merkle_root, transaction_count

                FROM `bigquery-public-data.crypto_bitcoin.blocks`

            )

            SELECT merkle_root, SUM(transaction_count) AS num_transactions

                FROM blocks

                GROUP BY merkle_root

                ORDER BY num_transactions DESC

        """



bitcoin_blockchain.estimate_query_size(query)
merkle_transactions = bitcoin_blockchain.query_to_pandas_safe(query)
merkle_transactions.head()
# Day 5 Joins

github = bq_helper.BigQueryHelper(active_project='bigquery-public-data',

                                  dataset_name='github_repos')

github.list_tables()
query = """

        -- Select all the columns we want in our joined table

        SELECT L.license, COUNT(sf.path) AS number_of_files

        FROM `bigquery-public-data.github_repos.sample_files` AS sf

        -- Table to merge into sample_files

        INNER JOIN `bigquery-public-data.github_repos.licenses` AS L

            ON sf.repo_name = L.repo_name -- the columns to join on

        GROUP BY L.license

        ORDER BY number_of_files DESC

        """

github.estimate_query_size(query)
file_count_by_license = github.query_to_pandas_safe(query, max_gb_scanned=6)

print(file_count_by_license)
github.head('sample_files', 5)
github.head('sample_commits', 1)
query = """ SELECT sf.repo_name, COUNT(sc.commit) AS file_count

                FROM `bigquery-public-data.github_repos.sample_commits` AS sc

                INNER JOIN `bigquery-public-data.github_repos.sample_files` AS sf 

                ON sf.repo_name = sc.repo_name

                WHERE sf.path LIKE '%.py'

                GROUP BY sf.repo_name

                ORDER BY file_count

            """

github.estimate_query_size(query)
commit_count = github.query_to_pandas_safe(query, max_gb_scanned=6)
commit_count.head()
import matplotlib.pyplot as plt



commit_count.plot()