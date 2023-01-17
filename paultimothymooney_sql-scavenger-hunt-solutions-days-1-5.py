import bq_helper
open_aq = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="openaq")
hacker_news = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="hacker_news")
accidents = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="nhtsa_traffic_fatalities")
bitcoin_blockchain = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                              dataset_name="bitcoin_blockchain")
github = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                              dataset_name="github_repos")
query1A = """SELECT distinct country
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE unit != 'ppm'
        """
response1A = open_aq.query_to_pandas_safe(query1A)
response1A.head(5)
query1B = """SELECT distinct pollutant
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE value = 0
        """
result1B = open_aq.query_to_pandas_safe(query1B)
result1B.head(5)
query2A = """
SELECT type,COUNT(id) AS count
FROM `bigquery-public-data.hacker_news.full`
GROUP BY type
ORDER BY count DESC
"""
result2A = hacker_news.query_to_pandas_safe(query2A)
result2A.head(5)
query2B = """
SELECT COUNT(deleted) AS Number_of_Deleted_Comments
FROM `bigquery-public-data.hacker_news.comments`
"""
result2B = hacker_news.query_to_pandas_safe(query2B)
result2B.head(5)
query2C = """SELECT author, 
            AVG(ranking)
            FROM `bigquery-public-data.hacker_news.comments`
            GROUP BY author
            ORDER BY AVG(ranking) DESC
        """
result2C = hacker_news.query_to_pandas_safe(query2C)
result2C.head(5)

query3A =     """ 
                                SELECT  DATE(timestamp_of_crash),
                                        EXTRACT(HOUR FROM timestamp_of_crash) AS `Hour`,
                                        COUNT(consecutive_number) AS `Accidents`
                                FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`
                                GROUP BY  timestamp_of_crash
                                ORDER BY  COUNT(consecutive_number) DESC                                        
                               """
result3A = accidents.query_to_pandas_safe(query3A)
result3A.head(10)
query3B =     """ 
                                SELECT registration_state_name AS `State`,
                                                COUNT(consecutive_number) AS `HitAndRuns`
                                FROM `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_2015`
                                WHERE hit_and_run LIKE "Yes" AND registration_state_name NOT LIKE "Unknown"
                                GROUP BY  registration_state_name
                                ORDER BY  COUNT(consecutive_number) DESC                                        
                               """
result3B = accidents.query_to_pandas_safe(query3B)
result3B.head(10)
query4A = """ WITH time AS 
            (
                SELECT TIMESTAMP_MILLIS(timestamp) AS trans_time,
                    transaction_id
                FROM `bigquery-public-data.bitcoin_blockchain.transactions`
            )
            SELECT COUNT(transaction_id) AS transactions,
                EXTRACT(DAY FROM trans_time) AS day,
                EXTRACT(MONTH FROM trans_time) AS month,
                EXTRACT(YEAR FROM trans_time) AS year
            FROM time
            WHERE EXTRACT(YEAR FROM trans_time) = 2017
            GROUP BY year, month, day 
            ORDER BY year, month, day
        """
result4A = bitcoin_blockchain.query_to_pandas_safe(query4A, max_gb_scanned=21)
print(result4A.head(5))
import matplotlib.pyplot as plt
plt.plot(result4A.transactions)
plt.title("Daily Bitcoin Transcations in 2017")
query4B="""SELECT COUNT(block_id) AS Number_of_Blocks, Merkle_Root
           FROM `bigquery-public-data.bitcoin_blockchain.blocks`
           GROUP BY Merkle_Root
           ORDER BY Number_of_Blocks
        """
result4B = bitcoin_blockchain.query_to_pandas(query4B)
result4B.head(10)
query5A = ("""
        SELECT COUNT(commit) AS Total_Number_of_Commits_in_Python
        FROM `bigquery-public-data.github_repos.sample_commits` AS sc
        INNER JOIN `bigquery-public-data.github_repos.sample_files` AS sf 
            ON sc.repo_name = sf.repo_name
            WHERE sf.path LIKE '%.py'        
        """)
result5A = github.query_to_pandas(query5A)
result5A.head(10)
query5B="""with temp AS
             (SELECT sf.path AS path, sc.commit AS commit, sc.repo_name AS Repository_Name
              FROM `bigquery-public-data.github_repos.sample_files` AS sf
              INNER JOIN `bigquery-public-data.github_repos.sample_commits` AS sc 
              ON sf.repo_name = sc.repo_name
              WHERE path LIKE '%.py'
            )
            SELECT count(commit) AS Number_of_Commits_in_Python, Repository_Name
            FROM temp
            GROUP BY Repository_Name 
            ORDER BY Number_of_Commits_in_Python DESC
            """
result5B = github.query_to_pandas(query5B)
result5B

