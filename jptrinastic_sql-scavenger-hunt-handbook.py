# import our bq_helper package
import bq_helper 
import matplotlib.pyplot as plt
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
hacker_news.head("full", selected_columns="by", num_rows=10)
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
#Load data using BigQuery
open_aq = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="openaq")
#Print tables in the dataset to choose from
print(open_aq.list_tables())
#View header of 'global_air_quality'
open_aq.head("global_air_quality")
#First query: note that backticks must be used around the FROM command
queryUnit = """SELECT country
               FROM `bigquery-public-data.openaq.global_air_quality`
               WHERE unit != 'ppm'
            """
dfNotPPM = open_aq.query_to_pandas_safe(queryUnit, max_gb_scanned=0.1)
#Use set for list of all countries not using ppm count)
set(dfNotPPM['country'])
#Second query: pollutants with exactly zero amount
queryUnit = """SELECT pollutant
               FROM `bigquery-public-data.openaq.global_air_quality`
               WHERE value = 0.0
            """
dfZeroValue = open_aq.query_to_pandas_safe(queryUnit, max_gb_scanned=0.1)
set(dfZeroValue['pollutant'])
#take a look at the hacker_news tables
hacker_news.list_tables()
#Query 1: how many stories are there of each type?
# print the first couple rows of the "full" table
hacker_news.head("full")
#query to count stories by type
queryStoriesCount = """
                    SELECT COUNT(id)
                    FROM `bigquery-public-data.hacker_news.full`
                    GROUP BY type"""
dfStoriesType = hacker_news.query_to_pandas_safe(queryStoriesCount, max_gb_scanned=0.3)
#Number of stories of each type
dfStoriesType
#Query 2: How many comments were deleted?
# print the first couple rows of the "comments" table
hacker_news.head("comments")
#First aggregate id using a count, then groupby deleted, then only keep those that are True
queryCommDel = """
               SELECT COUNT(id)
               FROM `bigquery-public-data.hacker_news.comments`
               GROUP BY deleted
               HAVING deleted=True"""
dfCommDel = hacker_news.query_to_pandas_safe(queryCommDel, max_gb_scanned=0.3)
#Number of deleted comments
dfCommDel
#Bonus: could also do average score by type of story
queryStoriesScore = """
                    SELECT AVG(score)
                    FROM `bigquery-public-data.hacker_news.full`
                    GROUP BY type"""
dfStoryScore = hacker_news.query_to_pandas_safe(queryStoriesScore, max_gb_scanned=0.3)
dfStoryScore
#Load US traffic fatality data
# create a helper object for this dataset
accidents = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="nhtsa_traffic_fatalities")
#Review available tables
accidents.list_tables()
#Examine 2015 data
accidents.head('accident_2015')
#Query 1: During which hours of the day do the most accidents occur in 2015?
#--Use extract and then choice of time type to take time value from datetime format
queryHours = """
             SELECT COUNT(consecutive_number),
                    EXTRACT(HOUR FROM timestamp_of_crash)
             FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`
             GROUP BY EXTRACT(HOUR FROM timestamp_of_crash)
             ORDER BY COUNT(consecutive_number) DESC
             """
dfHours = accidents.query_to_pandas_safe(queryHours, max_gb_scanned=0.3)
plt.figure()
plt.scatter(dfHours.f0_, dfHours.f1_)
#Query 2: Which state has the most hit-and-runs in 2015?
#Use the registration_state_name and hit_and_run columns
accidents.head('vehicle_2015').registration_state_name
queryHR = """
             SELECT COUNT(hit_and_run),
                    registration_state_name
             FROM `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_2015`
             GROUP BY registration_state_name
             ORDER BY COUNT(hit_and_run) DESC
             """
dfHR = accidents.query_to_pandas_safe(queryHR, max_gb_scanned=0.3)
#Texas, California, and  Florida are top three hit-and-run states
dfHR.head()
#Example query using WITH and AS to create a Common Table Expression (CTE), helpful for breaking up queries
bitcoinBlockchain = bq_helper.BigQueryHelper(active_project = 'bigquery-public-data',
                                             dataset_name = 'bitcoin_blockchain')
bitcoinBlockchain.head('transactions')
#Example query: Transactions per month
queryTransMonth = """
                  WITH time AS
                  (
                      SELECT TIMESTAMP_MILLIS(timestamp) AS trans_time,
                             transaction_id
                      FROM `bigquery-public-data.bitcoin_blockchain.transactions`
                  )
                  SELECT COUNT(transaction_id),
                         EXTRACT(MONTH FROM trans_time) AS month,
                         EXTRACT(YEAR FROM trans_time) AS year
                  FROM time
                  GROUP BY year, month
                  ORDER BY year, month"""
dfTransMonth = bitcoinBlockchain.query_to_pandas_safe(queryTransMonth, max_gb_scanned=0.3)
#Scavenger Hunt 1: How many bitcoin transactions were made each day in 2017?
query2017 = """
            WITH time AS
            (
                SELECT TIMESTAMP_MILLIS(timestamp) as trans_time,
                       transaction_id
                FROM `bigquery-public-data.bitcoin_blockchain.transactions`
            )
            SELECT COUNT(transaction_id) AS count,
                   EXTRACT(DAYOFYEAR FROM trans_time) AS day,
                   EXTRACT(YEAR FROM trans_time) AS year
            FROM time
            WHERE EXTRACT(YEAR FROM trans_time) = 2017
            GROUP BY day, year
            ORDER BY day, year"""
df2017 = bitcoinBlockchain.query_to_pandas_safe(query2017, max_gb_scanned=21.0)
plt.figure()
ax = plt.subplots(figsize=(10,5))
plt.plot(df2017['day'], df2017['count'])
#Scavenger Hunt 2: How many transactions are associated with each merkle root?
queryMerkle = """
              SELECT COUNT(transaction_id) AS count,
                     merkle_root
              FROM `bigquery-public-data.bitcoin_blockchain.transactions`
              GROUP BY merkle_root
              ORDER BY count DESC"""
dfMerkle = bitcoinBlockchain.query_to_pandas_safe(queryMerkle, max_gb_scanned=40.0)
dfMerkle.head()
github = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                  dataset_name="github_repos")
github.list_tables()
github.head('sample_files')
# Query: How many commits (recorded in the "sample_commits" table) have been made in repos written in the Python programming language?
# (I'm looking for the number of commits per repo for all the repos written in Python.
# Can determine python by %.py in 'path' in sample_files table

queryCommit = """
              --select repo name as total number of commits as quantities to return
              SELECT sf.repo_name, COUNT(sc.commit) AS num_commit
              FROM `bigquery-public-data.github_repos.sample_files` AS sf
              --join on the repo name (best common column I could find)
              INNER JOIN `bigquery-public-data.github_repos.sample_commits` AS sc
                  ON sf.repo_name = sc.repo_name
              --group by repo name to get total commits per repo
              GROUP BY sf.repo_name
              ORDER BY num_commit DESC"""
dfCommit = github.query_to_pandas_safe(queryCommit, max_gb_scanned=1.7)
dfCommit
