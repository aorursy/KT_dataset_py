import bq_helper
hacker_news = bq_helper.BigQueryHelper(active_project = "bigquery-public-data",
                                      dataset_name = "hacker_news")
hacker_news.list_tables()
hacker_news.table_schema("full")
hacker_news.head("full")
hacker_news.head("full", selected_columns = "by", num_rows = 10)
query = """SELECT score 
FROM `bigquery-public-data.hacker_news.full` 
WHERE type = "job" """
hacker_news.estimate_query_size(query)
hacker_news.query_to_pandas_safe(query, max_gb_scanned = 0.1)
job_post_scores = hacker_news.query_to_pandas_safe(query)
job_post_scores.score.mean()
job_post_scores.to_csv("job_post_scores.csv")
import bq_helper
open_aq = bq_helper.BigQueryHelper(active_project = 'bigquery-public-data',
                                  dataset_name = 'openaq')
open_aq.list_tables()
open_aq.head("global_air_quality")
query = """SELECT city
FROM `bigquery-public-data.openaq.global_air_quality`
WHERE country = 'US'
"""
us_cities = open_aq.query_to_pandas_safe(query)
us_cities.city.value_counts().head()
# Question 1 - Countries using anything other than ppm as unit of measurement
query1 = """SELECT distinct(country)
FROM `bigquery-public-data.openaq.global_air_quality`
WHERE unit != 'ppm'
"""
country_list = open_aq.query_to_pandas_safe(query1)
country_list.head()
country_list.shape
# Question 2 - I understand that this means which one pollutant has 0 value globally, meaning that 
# essentially that doesn't exist on the planet/ at the measurement sites. But this is not true
# for any pollutant. Switching to finding pollutants with value 0 anywhere.
query2 = """SELECT DISTINCT pollutant, value as `Pollutant_Level`
FROM `bigquery-public-data.openaq.global_air_quality`
WHERE value = 0
"""

pollutant_levels = open_aq.query_to_pandas_safe(query2)

pollutant_levels
import bq_helper
hacker_news = bq_helper.BigQueryHelper(active_project = "bigquery-public-data",
                                      dataset_name = "hacker_news")
hacker_news.head('comments')
query4 = """SELECT parent, COUNT(id)
FROM `bigquery-public-data.hacker_news.comments`
GROUP BY parent
HAVING COUNT(id) > 10
ORDER BY COUNT(id) DESC
"""
popular_stories = hacker_news.query_to_pandas_safe(query4)
popular_stories.head()
# Question 1 - How many stories are there of each type in full table?
query5 = """SELECT type, COUNT(id)
FROM `bigquery-public-data.hacker_news.full`
GROUP BY type
ORDER BY COUNT(id) DESC
"""
hacker_news.estimate_query_size(query5)
story_type = hacker_news.query_to_pandas_safe(query5)
story_type.shape
story_type.head()
# Question 2 - How many comments were deleted?
query6 = """SELECT COUNT(id)
FROM `bigquery-public-data.hacker_news.comments`
WHERE deleted = True
"""
hacker_news.estimate_query_size(query6)
deleted_comments = hacker_news.query_to_pandas_safe(query6)
deleted_comments
# Question 3 Alternative
query7 = """SELECT COUNTIF(deleted = True)
FROM `bigquery-public-data.hacker_news.comments`
"""
hacker_news.estimate_query_size(query7)
del_comments_alt = hacker_news.query_to_pandas_safe(query7)
del_comments_alt
import bq_helper
accidents = bq_helper.BigQueryHelper(active_project = "bigquery-public-data",
                                    dataset_name = "nhtsa_traffic_fatalities")
accidents.head('accident_2015')
query8 = """SELECT COUNT(consecutive_number),
EXTRACT(DAYOFWEEK FROM timestamp_of_crash)
FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`
GROUP BY EXTRACT(DAYOFWEEK FROM timestamp_of_crash)
ORDER BY COUNT(consecutive_number) DESC
"""
accidents.estimate_query_size(query8)
accident_by_weekday = accidents.query_to_pandas_safe(query8)
accident_by_weekday
import matplotlib.pyplot as plt
plt.scatter(x = accident_by_weekday.f1_,y = accident_by_weekday.f0_)
# Question 1 - Which hour of the day do most crashes happen at?
query9 = """SELECT COUNT(consecutive_number),
EXTRACT(HOUR FROM timestamp_of_crash)
FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`
GROUP BY EXTRACT(HOUR FROM timestamp_of_crash)
ORDER BY COUNT(consecutive_number) DESC
"""
accidents.estimate_query_size(query9)
accident_by_time = accidents.query_to_pandas_safe(query9)
accident_by_time
plt.scatter(x = accident_by_time.f1_, y = accident_by_time.f0_)
# Question 2 - Which state has the most hit and run cases?
query10 = """SELECT registration_state_name, COUNTIF(hit_and_run = "Yes")
FROM `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_2015`
GROUP BY registration_state_name
ORDER BY COUNTIF(hit_and_run = "Yes") DESC
"""

accidents.estimate_query_size(query10)
hit_and_run = accidents.query_to_pandas_safe(query10)
hit_and_run.head()
import bq_helper
bitcoin_blockchain = bq_helper.BigQueryHelper(active_project = "bigquery-public-data",
                                             dataset_name = "bitcoin_blockchain")
query11 = """WITH time AS
(
    SELECT TIMESTAMP_MILLIS(timestamp) AS trans_time,
    transaction_id
    FROM `bigquery-public-data.bitcoin_blockchain.transactions`
)
SELECT EXTRACT(YEAR FROM trans_time) AS Year,
EXTRACT(MONTH FROM trans_time) AS Month,
COUNT(transaction_id) AS transactions
FROM time
GROUP BY Year, Month
ORDER BY Year, Month 
"""

bitcoin_blockchain.estimate_query_size(query11)
trans_per_month = bitcoin_blockchain.query_to_pandas_safe(query11, max_gb_scanned = 21)
trans_per_month.head()
import matplotlib.pyplot as plt
plt.plot(trans_per_month.transactions)
# Question 1 - How many bitcoin transactions were made each day in 2017?

query12 = """WITH trans_all AS
(
    SELECT TIMESTAMP_MILLIS(timestamp) as Trans_Time,
    transaction_id
    FROM `bigquery-public-data.bitcoin_blockchain.transactions`
)
SELECT EXTRACT(DAYOFYEAR FROM Trans_Time) AS Day_Of_Year,
COUNT(transaction_id) as Number_Of_Transactions
FROM trans_all
WHERE EXTRACT(YEAR FROM Trans_Time) = 2017
GROUP BY Day_Of_Year
ORDER BY Day_Of_Year
"""
bitcoin_blockchain.estimate_query_size(query12)
trans_per_day_2017 = bitcoin_blockchain.query_to_pandas_safe(query12, max_gb_scanned = 21)
trans_per_day_2017.head()
plt.plot(trans_per_day_2017.Number_Of_Transactions)
plt.title("Number of Bitcoin Transactions Per Day in 2017")
plt.xlabel("Day Number")
# Question 2 - How many transactions are associated with each merkle root?
query13 = """SELECT merkle_root,
COUNT(transaction_id)
FROM `bigquery-public-data.bitcoin_blockchain.transactions`
GROUP BY merkle_root
ORDER BY COUNT(transaction_id) DESC
"""
bitcoin_blockchain.estimate_query_size(query13)
trans_per_merkle_root = bitcoin_blockchain.query_to_pandas_safe(query13, max_gb_scanned = 37)
trans_per_merkle_root.head()
import bq_helper
github = bq_helper.BigQueryHelper(active_project = 'bigquery-public-data',
                                 dataset_name = 'github_repos')
github.table_schema('licenses')
github.table_schema('sample_files')
query14 = """SELECT L.license, COUNT(sf.path) AS Number_Of_Files
FROM `bigquery-public-data.github_repos.sample_files` as sf
INNER JOIN `bigquery-public-data.github_repos.licenses` as L
ON sf.repo_name = L.repo_name
GROUP BY license
ORDER BY Number_Of_Files DESC
"""
github.estimate_query_size(query14)
file_count_by_license = github.query_to_pandas_safe(query14, max_gb_scanned = 6)
file_count_by_license.head()
# Question 1 - How many commits made in repos written in Python?
github.table_schema('sample_files')
github.table_schema('sample_commits')
query15 = """WITH sf_python AS
(
    SELECT DISTINCT repo_name
    FROM `bigquery-public-data.github_repos.sample_files`
    WHERE path LIKE '%.py'
)
SELECT sf.repo_name AS Repo, COUNT(sc.commit) AS Number_Of_Commits
FROM sf_python AS sf
INNER JOIN `bigquery-public-data.github_repos.sample_commits` AS sc
USING(repo_name)
GROUP BY Repo
ORDER BY Number_Of_Commits DESC
"""
github.estimate_query_size(query15)
commits_per_repo = github.query_to_pandas_safe(query15, max_gb_scanned = 6)
commits_per_repo.head()