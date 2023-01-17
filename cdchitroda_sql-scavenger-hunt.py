# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
# prints the current path
print (os.getcwd())
#prints the contents of the dir
os.listdir()
#path = '../input/'
#dirs = os.listdir(path)
#for file in dirs:
#    print (file)


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

#from subprocess import check_output
#print(check_output(["ls", "../input/"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
# import our bq_helper package
import bq_helper
# create a helper object for our bigquery dataset
hacker_news = bq_helper.BigQueryHelper(active_project= "bigquery-public-data", 
                                       dataset_name = "hacker_news")
# print a list of all the tables in the hacker_news dataset
hacker_news.list_tables()
# print information on all the columns in the "full" table
# in the hacker_news dataset
hacker_news.table_schema("full")
# preview the first couple lines of the "full" table
hacker_news.head("full")
# preview the first ten entries in the by column of the full table
hacker_news.head("full",selected_columns="by,score,type",num_rows=9)
#Finding the size of results a query will return
# this query looks in the full table in the hacker_news
# dataset, then gets the score column from every row where 
# the type column has "job" in it.
query = """SELECT score
            FROM `bigquery-public-data.hacker_news.full`
            WHERE type = "job" """

# check how big this query will be
hacker_news.estimate_query_size(query)

# only run this query if it's less than 100 MB
hacker_news.query_to_pandas_safe(query, max_gb_scanned=0.1)

# check out the scores of job postings (if the 
# query is smaller than 1 gig)
job_post_scores = hacker_news.query_to_pandas_safe(query)

# average score for job posts
job_post_scores.score.mean()
# save our dataframe as a .csv 
job_post_scores.to_csv("job_post_scores.csv")
open_aq = bq_helper.BigQueryHelper(active_project="bigquery-public-data",dataset_name="openaq")
open_aq.list_tables()
open_aq.head('global_air_quality')
query = """ SELECT city FROM
`bigquery-public-data.openaq.global_air_quality`
WHERE country = "US"
"""
open_aq.estimate_query_size(query)
us_cities = open_aq.query_to_pandas_safe(query=query,max_gb_scanned=0.1)
#us_cities.head()
us_cities.city.value_counts().head()
open_aq.head("global_air_quality")
query = """ SELECT country FROM `bigquery-public-data.openaq.global_air_quality`
WHERE unit != "ppm"
"""
open_aq.estimate_query_size(query)

unit_not_ppm = open_aq.query_to_pandas_safe(query=query,max_gb_scanned=0.1)
unit_not_ppm.country.value_counts().head()
open_aq.head("global_air_quality")
query = """ SELECT country FROM `bigquery-public-data.openaq.global_air_quality`
WHERE unit = "ppm"
"""

open_aq.estimate_query_size(query)
unit_ppm = open_aq.query_to_pandas_safe(query)
unit_ppm.country.value_counts().head()
query = """ SELECT pollutant FROM `bigquery-public-data.openaq.global_air_quality`
WHERE value = 0.00
"""

open_aq.estimate_query_size(query)

zero_poll = open_aq.query_to_pandas_safe(query)
zero_poll.pollutant.value_counts().head()
query = """ SELECT COUNT(country) 
FROM `bigquery-public-data.openaq.global_air_quality`
GROUP BY country
"""

open_aq.estimate_query_size(query)
import bq_helper
hack_news = bq_helper.BigQueryHelper(active_project="bigquery-public-data",dataset_name="hacker_news")
hack_news.list_tables()
hack_news.head(table_name="comments")
query = """ SELECT parent,COUNT(id)
FROM `bigquery-public-data.hacker_news.comments`
GROUP BY parent
HAVING COUNT(id) > 10
"""
hack_news.estimate_query_size(query)
pop_comments = hack_news.query_to_pandas_safe(query)
pop_comments.head()
hack_news.head("full")
query = """ SELECT type,COUNT(id)
FROM `bigquery-public-data.hacker_news.full`
GROUP BY type
HAVING COUNT(id) > 10
"""

hack_news.estimate_query_size(query)
hack_news.query_to_pandas_safe(query)
query = """ SELECT COUNT(type) as COUNT_type,MIN(score) as MIN_score,MAX(score) as MAX_score 
FROM `bigquery-public-data.hacker_news.full`
GROUP BY type
"""
hack_news.estimate_query_size(query)
hack_news.query_to_pandas_safe(query)
query = """ SELECT deleted,COUNT(id) as COUNT_deleted FROM `bigquery-public-data.hacker_news.full`
GROUP BY deleted
"""
hack_news.estimate_query_size(query)
hack_news.query_to_pandas_safe(query)
import bq_helper
accidents = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                     dataset_name="nhtsa_traffic_fatalities")
accidents.list_tables()

query = """ SELECT COUNT(consecutive_number),
EXTRACT(DAYOFWEEK FROM timestamp_of_crash) 
FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`
GROUP BY EXTRACT(DAYOFWEEK FROM timestamp_of_crash)
ORDER BY COUNT(consecutive_number) DESC
"""

accidents.estimate_query_size(query)
acc_by_day = accidents.query_to_pandas_safe(query)

#library for plotting
import matplotlib.pyplot as plt

# make a plot to show that our data is, actually, sorted:
plt.plot(acc_by_day.f0_)
plt.title("Number of accidents by rank \n (most to least dangerous)")
print(acc_by_day)
query = """ SELECT EXTRACT(HOUR FROM timestamp_of_crash),SUM(state_number)
FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`
GROUP BY EXTRACT(HOUR FROM timestamp_of_crash)
ORDER BY SUM(state_number) DESC
"""

accidents.estimate_query_size(query)
crash_hour = accidents.query_to_pandas_safe(query)
crash_hour
query = """ SELECT state_name,SUM(number_of_vehicle_forms_submitted_all)
FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`
GROUP BY state_name
ORDER BY SUM(number_of_vehicle_forms_submitted_all) DESC
"""

accidents.estimate_query_size(query)
accidents.query_to_pandas_safe(query)
query = """ SELECT first_harmful_event_name,COUNT(first_harmful_event)
FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`
GROUP BY first_harmful_event_name
ORDER BY COUNT(first_harmful_event) DESC
"""

accidents.estimate_query_size(query)
accidents.query_to_pandas_safe(query)
import bq_helper
btc_block = bq_helper.BigQueryHelper(active_project="bigquery-public-data",dataset_name="bitcoin_blockchain")
btc_block.list_tables()

# Using query with/as
query = """ WITH time as
 (
    SELECT TIMESTAMP_MILLIS(timestamp) as trans_time,
    transaction_id
    FROM `bigquery-public-data.bitcoin_blockchain.transactions`
 )
 SELECT COUNT(transaction_id) AS transactions,
 EXTRACT (MONTH FROM trans_time) AS month,
 EXTRACT (YEAR FROM trans_time) AS year
 FROM time
 GROUP BY year,month
 ORDER BY year,month
"""

transac_per_mth = btc_block.query_to_pandas_safe(query,max_gb_scanned=21)
#transac_per_mth.head()
import matplotlib.pyplot as plt

plt.plot(transac_per_mth.transactions)
plt.title("Monthly bitcoin transactions")
query = """ WITH trans AS
    (
     SELECT TIMESTAMP_MILLIS(timestamp) AS date_time,
     transaction_id as count_trans
     FROM `bigquery-public-data.bitcoin_blockchain.transactions`
    )
    SELECT EXTRACT(DAY FROM date_time) AS day,
    EXTRACT(MONTH FROM date_time) AS month,
    COUNT(count_trans) AS count_of_transactions
    FROM trans
    WHERE EXTRACT(YEAR FROM date_time) = 2017
    GROUP BY day,month
    ORDER BY day,month    
"""

btc_block.estimate_query_size(query)
per_day_trans = btc_block.query_to_pandas_safe(query,max_gb_scanned=21)
per_day_trans.tail()
per_day_trans.head()
plt.plot(per_day_trans.count_of_transactions)
plt.title("Daily BTC transactions 2017")
query = """ WITH merkle AS
 (
 SELECT merkle_root as root,
 transaction_id as trans
 FROM `bigquery-public-data.bitcoin_blockchain.transactions`
 )
 SELECT root,COUNT(trans) as trans_count
 FROM merkle
 GROUP BY root
 ORDER BY COUNT(trans) DESC
"""

btc_block.estimate_query_size(query)
merkle_count = btc_block.query_to_pandas_safe(query,max_gb_scanned=37)
merkle_count.tail()
import bq_helper
github = bq_helper.BigQueryHelper(active_project="bigquery-public-data",dataset_name="github_repos")
github.list_tables()
github.table_schema("sample_files")
github.head("sample_files")
query = ("""
        -- Select all the columns we want in our joined table
        SELECT L.license, COUNT(sf.path) AS number_of_files
        FROM `bigquery-public-data.github_repos.sample_files` as sf
        -- Table to merge into sample_files
        INNER JOIN `bigquery-public-data.github_repos.licenses` as L 
            ON sf.repo_name = L.repo_name -- what columns should we join on?
        GROUP BY L.license
        ORDER BY number_of_files DESC
        """)


github.estimate_query_size(query)

github.table_schema("sample_commits")
github.head("sample_commits")
query = """ SELECT COUNT(commit) as count_commit,sc.repo_name
FROM `bigquery-public-data.github_repos.sample_commits` AS sc
INNER JOIN `bigquery-public-data.github_repos.sample_files` AS sf
ON sc.repo_name = sf.repo_name
WHERE sf.path LIKE '%.py' 
GROUP BY repo_name
ORDER BY COUNT(commit) DESC
"""

github.estimate_query_size(query)
commits_python = github.query_to_pandas_safe(query,max_gb_scanned=6)
commits_python