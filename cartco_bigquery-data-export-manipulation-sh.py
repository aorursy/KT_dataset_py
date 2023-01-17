# Import BigQuery Helper library
import bq_helper
# Create BQ Helper object for the data
open_aq = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="openaq")
# Query the table for countries and unit, where unit is not equal to ppm
query = """SELECT DISTINCT country, unit
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE unit != 'ppm'
        """
# Run the query using safe function
unit_subset = open_aq.query_to_pandas_safe(query)
# Print the results
print(unit_subset)
#unit_subset.to_csv("output1")
# Query the table for pollutants and value, where value is equal to 0
query2 = """SELECT DISTINCT pollutant, value
             FROM `bigquery-public-data.openaq.global_air_quality`
             WHERE value = 0.00
         """
nil_value = open_aq.query_to_pandas_safe(query2)
# Print the results
print(nil_value)
#nil_value.to_csv("output2")
# Create a BQ Helper object for this dataset
hacker_news = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="hacker_news")

# List the tables within the dataset
hacker_news.list_tables()
# Print the first couple rows of the "comments" table
hacker_news.head("comments")

# Query for comments with atleast ten replies
query = """SELECT parent, COUNT(id)
            FROM `bigquery-public-data.hacker_news.comments`
            GROUP BY parent
            HAVING COUNT(id) > 10
        """

# Run the query with a limit of 1 gb of data scanned
tenrep = hacker_news.query_to_pandas_safe(query)

tenrep.head()
hacker_news.head("full")

query = """SELECT type, COUNT(id)
            FROM `bigquery-public-data.hacker_news.full`
            GROUP BY type
        """
stories = hacker_news.query_to_pandas_safe(query)
print(stories)

#  Following the tutorial
query = """SELECT COUNT(id)
            FROM `bigquery-public-data.hacker_news.full`
            GROUP BY deleted
            HAVING deleted = True
        """
hacker_news.estimate_query_size(query)
deleted_comments = hacker_news.query_to_pandas_safe(query)
print(deleted_comments)

# better code
query2 = """SELECT COUNT(id)
             FROM `bigquery-public-data.hacker_news.full`
             WHERE deleted = True
         """
deleted_comments2 = hacker_news.query_to_pandas_safe(query2)
print(deleted_comments2)

# most succinct code
query3 = """SELECT COUNTIF(deleted=True)
             FROM `bigquery-public-data.hacker_news.full`
         """
deleted_comments3 = hacker_news.query_to_pandas_safe(query3)
print(deleted_comments3)
# create a helper object for this dataset
accidents = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="nhtsa_traffic_fatalities")

query = """SELECT COUNT(consecutive_number) AS FREQUENCY,
                  EXTRACT(HOUR FROM timestamp_of_crash) AS HOUR
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`
            GROUP BY HOUR
            ORDER BY FREQUENCY DESC
        """
# Create a dataframe based on the query
accidents_by_hour = accidents.query_to_pandas_safe(query)
# Print the dataframe created by the query
print(accidents_by_hour)

import matplotlib.pyplot as plt

# Note: the data is sorted and the x axis does not represent the hour of day
plt.plot(accidents_by_hour.FREQUENCY)
plt.title("Number of Accidents by Hour \n (Most to least dangerous)")
query = """SELECT COUNT(vehicle_number), registration_state_name
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_2015`
            WHERE hit_and_run = 'Yes'
            GROUP BY registration_state_name
            ORDER BY COUNT(vehicle_number) DESC
        """
hnr_by_state = accidents.query_to_pandas_safe(query)
print(hnr_by_state)
bitcoin_blockchain = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                  dataset_name="bitcoin_blockchain")

# Two part query
# - Convert the timestamp
# - Get information 
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

# Increase max gb scanned to 21
transactions_per_month = bitcoin_blockchain.query_to_pandas_safe(query, max_gb_scanned=22)

# plot monthly bitcoin transactions
plt.plot(transactions_per_month.transactions)
plt.title("Monthly Bitcoin Transactions")
query = """ 
            WITH time AS 
                (
                    SELECT TIMESTAMP_MILLIS(timestamp) AS trans_time,
                    EXTRACT(YEAR FROM TIMESTAMP_MILLIS(timestamp)) AS year,
                        transaction_id
                    FROM `bigquery-public-data.bitcoin_blockchain.transactions`
                    WHERE EXTRACT(YEAR FROM TIMESTAMP_MILLIS(timestamp)) = 2017
                )
            SELECT EXTRACT(DAYOFYEAR FROM trans_time) AS Day_of_the_Year,
                    COUNT(transaction_id) AS transactions
            FROM time
            GROUP BY Day_of_the_Year
            ORDER BY Day_of_the_Year            
        """

transactions_per_day = bitcoin_blockchain.query_to_pandas_safe(query, max_gb_scanned=22)
# Print the number of Bitcoin transactions for the first 20 days of 2017
print(transactions_per_day.head(20))


# plot daily bitcoin transactions
plt.plot(transactions_per_day.transactions)
plt.title("Daily Bitcoin Transactions")
plt.xlabel('Day of the Year')
plt.ylabel('Number of Transactions')

# Common table expression not necessary in this query, using it for practice.
query = """WITH merkle AS
                (
                SELECT merkle_root, transaction_id
                FROM `bigquery-public-data.bitcoin_blockchain.transactions`
                )
            SELECT merkle_root, COUNT(transaction_id) AS transactions
            FROM merkle
            GROUP BY merkle_root
            ORDER BY transactions DESC
        """

bitcoin_blockchain.estimate_query_size(query)

transactions_per_merkle_root = bitcoin_blockchain.query_to_pandas_safe(query, max_gb_scanned=38)

# How many merkle roots are in this dataset?
print(len(transactions_per_merkle_root))
# How many transactions per merkle root in the top twenty five roots?
print(transactions_per_merkle_root.head(25))
# How many transactions are in the dataset
sum(transactions_per_merkle_root["transactions"])