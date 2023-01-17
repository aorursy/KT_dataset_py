# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# import package with helper functions 
import bq_helper

# create a helper object for this dataset
open_aq = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="openaq")

# print all the tables in this dataset (there's only one!)
open_aq.list_tables()

# Any results you write to the current directory are saved as output.
open_aq.head("global_air_quality")
open_aq.table_schema("global_air_quality")
query = """SELECT unit
            FROM `bigquery-public-data.openaq.global_air_quality`
        """
unit = open_aq.query_to_pandas_safe(query)
unit.unit.value_counts().head()
query = """SELECT DISTINCT country, unit
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE unit != 'ppm'
        """
country_not_ppm = open_aq.query_to_pandas_safe(query)
country_not_ppm
query = """SELECT DISTINCT pollutant, value
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE value = 0
        """
zero_pollution = open_aq.query_to_pandas_safe(query)
zero_pollution
# create a helper object for this dataset
hacker_news = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="hacker_news")

# print all the tables in this dataset (there's only one!)
hacker_news.list_tables()
hacker_news.table_schema("full")
hacker_news.head("full")
query = """SELECT type, COUNT(id)
            FROM `bigquery-public-data.hacker_news.full`
            GROUP BY type
        """
count = hacker_news.query_to_pandas_safe(query)

count
hacker_news.head("comments")
hacker_news.table_schema("comments")
query = """SELECT COUNT(id) as count
            FROM `bigquery-public-data.hacker_news.comments`
            WHERE deleted = True
        """
deleted_comment = hacker_news.query_to_pandas_safe(query)
deleted_comment
query = """SELECT `by`, type, COUNT(id) as count
            FROM `bigquery-public-data.hacker_news.full`
            WHERE `by` != ''
            GROUP BY `by`, type
            ORDER BY type, count DESC
        """
count2 = hacker_news.query_to_pandas_safe(query)
count2.head
accidents = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="nhtsa_traffic_fatalities")
accidents.list_tables()
accidents.head("accident_2015")
accidents.table_schema("accident_2015")
query = """SELECT EXTRACT(HOUR FROM timestamp_of_crash) as hour, COUNT(consecutive_number) as count
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`
            GROUP BY hour
            ORDER BY count DESC
        """
accident_hours = accidents.query_to_pandas_safe(query)
accident_hours
accidents.head("vehicle_2015")
accidents.table_schema("vehicle_2015")
query = """SELECT registration_state_name, COUNT(vehicle_number) as count
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_2015`
            WHERE hit_and_run = 'Yes'
            GROUP BY registration_state_name
            ORDER BY count DESC
        """
vehicle = accidents.query_to_pandas_safe(query)
vehicle
btc = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                              dataset_name="bitcoin_blockchain")
btc.head("transactions")
btc.table_schema("transactions")
query = """ WITH time as 
            (
                SELECT TIMESTAMP_MILLIS(timestamp) AS transaction_time, transaction_id
                FROM `bigquery-public-data.bitcoin_blockchain.transactions`
            )
            SELECT EXTRACT(DAY FROM transaction_time) as day, 
                EXTRACT(MONTH FROM transaction_time) as month, 
                COUNT(transaction_id) as count
                
            FROM time
            WHERE EXTRACT(year FROM transaction_time) = 2017
            GROUP BY day, month
            ORDER BY month, day
        """
# note that max_gb_scanned is set to 21, rather than 1
transactions_per_day = btc.query_to_pandas_safe(query, max_gb_scanned=21)
transactions_per_day
print(transactions_per_day.count)
query = """ SELECT merkle_root, COUNT(transaction_id) as total_transaction
            FROM `bigquery-public-data.bitcoin_blockchain.transactions`
            GROUP BY merkle_root
            ORDER BY total_transaction DESC
        """
# note that max_gb_scanned is set to 21, rather than 1
transactions_merkle_root = btc.query_to_pandas_safe(query, max_gb_scanned=37)
transactions_merkle_root
github = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                              dataset_name="github_repos")
query = """ SELECT li.license, COUNT(id) as total
            FROM `bigquery-public-data.github_repos.sample_files` as sf
            INNER JOIN `bigquery-public-data.github_repos.licenses` as li 
            ON sf.repo_name = li.repo_name 
            GROUP BY li.license
            ORDER BY total DESC
        """
# note that max_gb_scanned is set to 21, rather than 1
license = github.query_to_pandas_safe(query, max_gb_scanned=5)
license
query = """ SELECT sc.repo_name, COUNT(DISTINCT sc.commit) as total_commit
            FROM `bigquery-public-data.github_repos.sample_commits` as sc
            INNER JOIN `bigquery-public-data.github_repos.sample_files` as sf 
            ON sf.repo_name = sc.repo_name 
            WHERE sf.path LIKE "%.py"
            GROUP BY sc.repo_name
            ORDER BY total_commit DESC
        """
# note that max_gb_scanned is set to 21, rather than 1
python = github.query_to_pandas_safe(query, max_gb_scanned=6)
python
query = """ WITH python_repo AS 
            (
                SELECT DISTINCT sf.repo_name as repo_name
                FROM `bigquery-public-data.github_repos.sample_files` as sf 
                WHERE sf.path LIKE '%.py'
            )
            SELECT sc.repo_name, COUNT(sc.commit) as total
            FROM python_repo pr
            INNER JOIN `bigquery-public-data.github_repos.sample_commits` as sc
            ON pr.repo_name = sc.repo_name
            GROUP BY sc.repo_name
            ORDER BY total DESC
        """
# note that max_gb_scanned is set to 21, rather than 1
python2 = github.query_to_pandas_safe(query, max_gb_scanned=6)
python2