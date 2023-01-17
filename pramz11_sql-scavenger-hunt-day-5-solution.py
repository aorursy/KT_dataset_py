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
print(file_count_by_license)
# How many commits have been made in repos written in Python 

query = ("""
         -- select all the columns that have to be in the joined table 
         SELECT sf.repo_name, COUNT (c.commit) AS commits_in_python
         FROM `bigquery-public-data.github_repos.sample_files` AS sf
        
         -- Table to mererge into sample_files
         INNER JOIN `bigquery-public-data.github_repos.sample_commits` AS c
             ON sf.repo_name = c.repo_name
         WHERE sf.path LIKE '%.py'
         -- WHERE clause has to come after the join is made
         GROUP BY sf.repo_name
         ORDER BY commits_in_python DESC
         
        """)

# Check query size
github.estimate_query_size(query)
# Set max_gb_scanned limit to 6
commits_in_py = github.query_to_pandas_safe(query, max_gb_scanned = 6)
commits_in_py
# How many commits have been made in repos written in Python 
# Eliminate duplicate values using distinct statement 

query1 = """
              -- select all the distinct python repos
             WITH python_repos AS (
              SELECT DISTINCT repo_name
              FROM `bigquery-public-data.github_repos.sample_files` 
              WHERE path LIKE '%.py')
     
            SELECT c.repo_name, COUNT(commit) AS commits_in_python
            FROM `bigquery-public-data.github_repos.sample_commits` AS c
            JOIN python_repos
              ON python_repos.repo_name = c.repo_name
            GROUP BY c.repo_name
            ORDER BY commits_in_python DESC
    
         
        """

# Check query size
github.estimate_query_size(query1)
# Set max_gb_scanned limit to 6
commits_in_py = github.query_to_pandas_safe(query1, max_gb_scanned = 6)
# Results 
commits_in_py