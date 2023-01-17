import numpy as np
import pandas as pd
from bq_helper import BigQueryHelper
import matplotlib.pyplot as plt
import seaborn as sns
openaq = BigQueryHelper("bigquery-public-data", "openaq")
openaq.head("global_air_quality")
non_ppm_countries = openaq.query_to_pandas_safe("""
    SELECT DISTINCT country
    FROM   `bigquery-public-data.openaq.global_air_quality`
    WHERE  unit != 'ppm'
""")
" ".join(sorted(non_ppm_countries.country))
zero_pollutants = openaq.query_to_pandas_safe("""
    SELECT DISTINCT pollutant
    FROM   `bigquery-public-data.openaq.global_air_quality`
    WHERE  value = 0
""")
sorted(zero_pollutants.pollutant)
hacker_news = BigQueryHelper("bigquery-public-data", "hacker_news")
hacker_news.head("comments")
popular_comments = hacker_news.query_to_pandas_safe("""
    SELECT parent, COUNT(id) AS num_replies
    FROM `bigquery-public-data.hacker_news.comments`
    GROUP BY parent
    HAVING COUNT(id) > 10
    ORDER BY 2 DESC
""")
popular_comments.head()
stories_by_type = hacker_news.query_to_pandas_safe("""
    SELECT type, COUNT(id) AS num_of_type
    FROM `bigquery-public-data.hacker_news.full`
    GROUP BY type
""")
stories_by_type
deleted_comments = hacker_news.query_to_pandas_safe("""
    SELECT COUNT(id) AS num_deleted
    FROM `bigquery-public-data.hacker_news.comments`
    WHERE deleted = True
""")
deleted_comments
accidents = BigQueryHelper("bigquery-public-data", "nhtsa_traffic_fatalities")
print(*accidents.list_tables())
accidents.head("accident_2015", selected_columns="consecutive_number timestamp_of_crash year_of_crash day_of_week hour_of_crash minute_of_crash".split())
accidents_by_day = accidents.query_to_pandas_safe("""
    SELECT COUNT(consecutive_number) AS crash_count,
           EXTRACT(DAYOFWEEK FROM timestamp_of_crash) AS week_day
    FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`
    GROUP BY week_day
    ORDER BY crash_count DESC
""")
accidents_by_day
import matplotlib.pyplot as plt
plt.plot(accidents_by_day.crash_count)
_ = plt.title("Accidents by day")
accidents_by_hour = accidents.query_to_pandas_safe("""
    SELECT EXTRACT(HOUR FROM timestamp_of_crash) AS hour,
           COUNT(consecutive_number) AS number
    FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`
    GROUP BY hour
    ORDER BY number DESC
""")
accidents_by_hour
hit_and_run = accidents.query_to_pandas_safe("""
    SELECT registration_state_name AS reg_state,
           COUNT(consecutive_number) AS number
    FROM `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_2016`
    WHERE hit_and_run = "Yes"
    GROUP BY reg_state
    ORDER BY number DESC
""")
hit_and_run.head()
bitcoin_blockchain = BigQueryHelper("bigquery-public-data", "bitcoin_blockchain")
trans_by_month = bitcoin_blockchain.query_to_pandas_safe("""
    WITH TransTime AS (
        SELECT transaction_id AS trans_id,
               TIMESTAMP_MILLIS(timestamp) AS trans_time
        FROM `bigquery-public-data.bitcoin_blockchain.transactions`
    )
    SELECT COUNT(trans_id) AS number,
           EXTRACT(month FROM trans_time) AS month,
           EXTRACT(year FROM trans_time) AS year
    FROM TransTime
    GROUP BY year, month
    ORDER BY year, month
""", max_gb_scanned=21)
trans_by_month
_ = plt.plot(trans_by_month.number)
trans_by_day_2017 = bitcoin_blockchain.query_to_pandas_safe("""
    WITH TransTime AS (
        SELECT transaction_id AS trans_id,
               TIMESTAMP_MILLIS(timestamp) AS trans_time
        FROM `bigquery-public-data.bitcoin_blockchain.transactions`
    )
    SELECT COUNT(trans_id) AS number,
           EXTRACT(month FROM trans_time) AS month,
           EXTRACT(day FROM trans_time) AS day
    FROM TransTime
    WHERE EXTRACT(year FROM trans_time) = 2017
    GROUP BY month, day
    ORDER BY month, day
""", max_gb_scanned=21)
trans_by_day_2017
_ = plt.plot(trans_by_day_2017.number)
trans_by_merkle_root = bitcoin_blockchain.query_to_pandas_safe("""
    SELECT merkle_root,
           COUNT(DISTINCT transaction_id) AS number
    FROM `bigquery-public-data.bitcoin_blockchain.transactions`
    GROUP BY merkle_root
    ORDER BY number DESC
""", max_gb_scanned=37)
trans_by_merkle_root.head(10)
github = BigQueryHelper("bigquery-public-data", "github_repos")
query = """
    SELECT L.license, COUNT(F.path) AS num_files
    FROM `bigquery-public-data.github_repos.licenses` as L
    JOIN `bigquery-public-data.github_repos.sample_files` as F
    ON L.repo_name = F.repo_name
    GROUP BY license
    ORDER BY num_files DESC
"""
github.estimate_query_size(query)
github_licenses = github.query_to_pandas_safe(query, max_gb_scanned=6)
github_licenses
query = """
    SELECT C.repo_name, COUNT(C.commit) AS num_commits
    FROM `bigquery-public-data.github_repos.sample_commits` as C
    JOIN `bigquery-public-data.github_repos.sample_files` as F
    ON C.repo_name = F.repo_name
    WHERE F.path LIKE '%.py'
    GROUP BY repo_name
    ORDER BY num_commits DESC
"""
github.estimate_query_size(query)
github.query_to_pandas_safe(query, max_gb_scanned=5.5)
query = """
    SELECT C.repo_name, COUNT(DISTINCT C.commit) AS num_commits
    FROM `bigquery-public-data.github_repos.sample_commits` as C
    JOIN `bigquery-public-data.github_repos.sample_files` as F
    ON C.repo_name = F.repo_name
    WHERE F.path LIKE '%.py'
    GROUP BY repo_name
    ORDER BY num_commits DESC
"""
print(github.estimate_query_size(query))
github.query_to_pandas_safe(query, max_gb_scanned=5.5)
query = """
    SELECT repo_name, COUNT(commit) AS num_commits
    FROM `bigquery-public-data.github_repos.sample_commits`
    WHERE repo_name IN (
        SELECT repo_name
        FROM `bigquery-public-data.github_repos.sample_files`
        WHERE path LIKE '%.py'
    )
    GROUP BY repo_name
    ORDER BY num_commits DESC
"""
print(github.estimate_query_size(query))
github.query_to_pandas_safe(query, max_gb_scanned=5.5)
query = """
    WITH PythonRepos AS (
        SELECT DISTINCT repo_name
        FROM `bigquery-public-data.github_repos.sample_files`
        WHERE path LIKE '%.py'
    )
    SELECT C.repo_name, COUNT(C.commit) AS num_commits
    FROM `bigquery-public-data.github_repos.sample_commits` C
    JOIN PythonRepos R ON C.repo_name = R.repo_name
    GROUP BY repo_name
    ORDER BY num_commits DESC
"""
print(github.estimate_query_size(query))
github.query_to_pandas_safe(query, max_gb_scanned=5.5)


