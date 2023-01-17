# import our bq_helper package
import bq_helper 

# create a helper object for our bigquery dataset
bitcoin_blockchain = bq_helper.BigQueryHelper(active_project= "bigquery-public-data", 
                                       dataset_name = "bitcoin_blockchain")

# print a list of all the tables in the bitcoin_blockchain dataset
bitcoin_blockchain.list_tables()
# print information on all the columns in the "full" table
# in the bitcoin_blockchain dataset
bitcoin_blockchain.table_schema("blocks")
# preview the first couple lines of the "blocks" table
bitcoin_blockchain.head("blocks")
# preview the first ten entries in the block_id column of the full table
bitcoin_blockchain.head("blocks", selected_columns="block_id", num_rows=10)
# this query looks in the blocks table in the bitcoin_blockchain
# dataset, then gets the block_id column from every row where 
# the difficultyTarget column is greater than 453179945.
query = """SELECT block_id
            FROM `bigquery-public-data.bitcoin_blockchain.blocks`
            WHERE difficultyTarget > 453179945 """

# check how big this query will be
bitcoin_blockchain.estimate_query_size(query)
# only run this query if it's less than 30 MB
bitcoin_blockchain.query_to_pandas_safe(query, max_gb_scanned=0.03)
# (if the query is smaller than 1 gig) returns a Pandas dataframe
blockid_highdiff = bitcoin_blockchain.query_to_pandas_safe(query)

# average score for job posts
blockid_highdiff.head()
# save our dataframe as a .csv 
blockid_highdiff.to_csv("blockid_highdiff.csv")
# import package with helper functions 
import bq_helper

# create a helper object for this dataset
open_aq = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="openaq")

# print all the tables in this dataset (there's only one!)
open_aq.list_tables()
# print the first couple rows of the "global_air_quality" dataset
open_aq.head("global_air_quality")
# query to select all the items from the "city" column where the
# "country" column is "us"
query = """SELECT city
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE country = 'US'
        """
# the query_to_pandas_safe will only return a result if it's less
# than one gigabyte (by default)
us_cities = open_aq.query_to_pandas_safe(query)

# What five cities have the most measurements taken there?
us_cities.city.value_counts().head()
# Which countries use a unit other than ppm to measure any type of pollution? 
# (Hint: to get rows where the value isn't something, use "!=")

query = """SELECT country
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE unit != 'ppm'
        """

# Countries that have different units
countries_diff_units = open_aq.query_to_pandas_safe(query)

countries_diff_units.country.value_counts()
# Which pollutants have a value of exactly 0?

query = """SELECT pollutant
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE value = 0
        """

# Countries that have different units
zero_pollutant = open_aq.query_to_pandas_safe(query)

zero_pollutant.pollutant.value_counts()
# import package with helper functions 
import bq_helper

# create a helper object for this dataset
hacker_news = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="hacker_news")
hacker_news.list_tables()
# print the first couple rows of the "comments" table
hacker_news.head("comments")
# query to pass to 
query = """SELECT parent, COUNT(id)
            FROM `bigquery-public-data.hacker_news.comments`
            GROUP BY parent
            HAVING COUNT(id) > 10
        """

# the query_to_pandas_safe method will cancel the query if
# it would use too much of your quota, with the limit set 
# to 1 GB by default
popular_stories = hacker_news.query_to_pandas_safe(query)


popular_stories.head()
# need to see what tables are in the dataset
hacker_news.list_tables()
# print the first couple rows of the "full" table
hacker_news.head("full")

query = """SELECT type, COUNT(id)
            FROM `bigquery-public-data.hacker_news.full`
            GROUP BY type
            ORDER BY type
        """
type_of_stories = hacker_news.query_to_pandas_safe(query)
type_of_stories
query = """SELECT deleted, COUNT(id)
            FROM `bigquery-public-data.hacker_news.comments`
            GROUP BY deleted
            HAVING deleted = True
        """
deleted_comments = hacker_news.query_to_pandas_safe(query)
deleted_comments
query = """SELECT author, AVG(score)
            FROM `bigquery-public-data.hacker_news.stories`
            GROUP BY author
            ORDER BY AVG(score) DESC
        """
highest_score = hacker_news.query_to_pandas_safe(query)
highest_score.head(5)

# import package with helper functions 
import bq_helper

# create a helper object for this dataset
accidents = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="nhtsa_traffic_fatalities")

# query to find out the number of accidents which 
# happen on each day of the week
query = """SELECT COUNT(consecutive_number), 
                  EXTRACT(DAYOFWEEK FROM timestamp_of_crash)
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`
            GROUP BY EXTRACT(DAYOFWEEK FROM timestamp_of_crash)
            ORDER BY COUNT(consecutive_number) DESC
        """

# the query_to_pandas_safe method will cancel the query if
# it would use too much of your quota, with the limit set 
# to 1 GB by default
accidents_by_day = accidents.query_to_pandas_safe(query)

print(accidents_by_day)
# library for plotting
import matplotlib.pyplot as plt

# make a plot to show that our data is, actually, sorted:
plt.plot(accidents_by_day.f0_)
plt.title("Number of Accidents by Rank of Day \n (Most to least dangerous)")

# query to find out the number of accidents which 
# happen on each hour of the day
query = """SELECT COUNT(consecutive_number), 
                  EXTRACT(HOUR FROM timestamp_of_crash)
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`
            GROUP BY EXTRACT(HOUR FROM timestamp_of_crash)
            ORDER BY COUNT(consecutive_number) DESC
        """

# the query_to_pandas_safe method will cancel the query if
# it would use too much of your quota, with the limit set 
# to 1 GB by default
accidents_by_hour = accidents.query_to_pandas_safe(query)

print(accidents_by_hour)
# query to find out which states had the highest number of hit and runs
query = """SELECT registration_state_name, COUNT(hit_and_run) 
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_2015`
            GROUP BY registration_state_name
            ORDER BY COUNT(hit_and_run) DESC
        """

# the query_to_pandas_safe method will cancel the query if
# it would use too much of your quota, with the limit set 
# to 1 GB by default
state_hit = accidents.query_to_pandas_safe(query)

state_hit.head(1)
# import package with helper functions 
import bq_helper

# create a helper object for this dataset
bitcoin_blockchain = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                              dataset_name="bitcoin_blockchain")

# converting the integer to a timestamp
# get information on the date of transactions from the timestamp
query = """ WITH time AS 
            (
                SELECT TIMESTAMP_MILLIS(timestamp) AS trans_time,
                    transaction_id
                FROM `bigquery-public-data.bitcoin_blockchain.transactions`
            )
            SELECT COUNT(transaction_id) AS transactions,
                EXTRACT(MONTH FROM trans_time) AS month,
                EXTRACT(YEAR FROM trans_time) AS year
            FROM time
            GROUP BY year, month 
            ORDER BY year, month
        """

# note that max_gb_scanned is set to 21, rather than 1
transactions_per_month = bitcoin_blockchain.query_to_pandas_safe(query, max_gb_scanned=21)

# import plotting library
import matplotlib.pyplot as plt

# plot monthly bitcoin transactions
plt.plot(transactions_per_month.transactions)
plt.title("Monthly Bitcoin Transcations")
# converting the integer to a timestamp
# get information on the date of transactions from the timestamp
query = """ WITH time AS 
            (
                SELECT TIMESTAMP_MILLIS(timestamp) AS trans_time,
                    transaction_id
                FROM `bigquery-public-data.bitcoin_blockchain.transactions`
            )
            SELECT EXTRACT(MONTH FROM trans_time) AS month,
                EXTRACT(DAY FROM trans_time) AS day,
                COUNT(transaction_id) AS transactions
            FROM time
            WHERE EXTRACT(YEAR FROM trans_time)=2017
            GROUP BY month, day 
            ORDER BY month, day
        """

# note that max_gb_scanned is set to 21, rather than 1
transactions_per_day = bitcoin_blockchain.query_to_pandas_safe(query, max_gb_scanned=21)

print(transactions_per_day.head())

# plot daily bitcoin transactions
plt.plot(transactions_per_day.transactions)
plt.title("2017 Daily Bitcoin Transcations")
query = """ SELECT merkle_root,
                COUNT(transaction_id) AS number_of_transactions
            FROM `bigquery-public-data.bitcoin_blockchain.transactions`
            GROUP BY merkle_root 
            ORDER BY COUNT(transaction_id) DESC
        """

# note that max_gb_scanned is set to 38, rather than 25
merkle_transactions = bitcoin_blockchain.query_to_pandas_safe(query, max_gb_scanned=38)

merkle_transactions.head(5)
# import package with helper functions 
import bq_helper

# create a helper object for this dataset
github = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                              dataset_name="github_repos")


# You can use two dashes (--) to add comments in SQL
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

file_count_by_license = github.query_to_pandas_safe(query, max_gb_scanned=6)

# print out all the returned results
file_count_by_license
# query provides number of commits in Python per repository
query = (""" SELECT sf.repo_name, COUNT(sc.commit) AS number_of_commits
        FROM `bigquery-public-data.github_repos.sample_commits` as sc
        INNER JOIN `bigquery-public-data.github_repos.sample_files` as sf 
            ON sf.repo_name = sc.repo_name 
        WHERE sf.path LIKE '%.py'
        GROUP BY sf.repo_name
        ORDER BY number_of_commits DESC
        """)

commits_in_repos = github.query_to_pandas_safe(query, max_gb_scanned=6)

# print out all the returned results
commits_in_repos
# query provides the total number of of commits that have been made in 
# repositories written in Python
query = (""" SELECT COUNT(sc.commit) AS total_number_of_commits
        FROM `bigquery-public-data.github_repos.sample_commits` as sc
        INNER JOIN `bigquery-public-data.github_repos.sample_files` as sf 
            ON sf.repo_name = sc.repo_name 
        WHERE sf.path LIKE '%.py'
        """)

total_commits = github.query_to_pandas_safe(query, max_gb_scanned=6)

# print out all the returned results
total_commits
