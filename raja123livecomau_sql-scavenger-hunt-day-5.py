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
# Below code display Duplicate rows for rep names commited as well as unique rows :)
# import package with helper functions 
import bq_helper

# create a helper object for this dataset
github = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                              dataset_name="github_repos")
# query with duplicate repo names in the sample file

query1 = ("""
        -- Select all the columns we want in our joined table
        SELECT sc.repo_name, COUNT(sc.commit) AS number_of_commits
        FROM `bigquery-public-data.github_repos.sample_commits` as sc
        -- Table to merge into sample_files
        INNER JOIN `bigquery-public-data.github_repos.sample_files` as sf 
            ON sf.repo_name = sc.repo_name -- what columns should we join on?
        WHERE sf.path LIKE ('%.py')
        GROUP BY sc.repo_name
        ORDER BY number_of_commits DESC
        """)

file_count_by_repo1 = github.query_to_pandas_safe(query1, max_gb_scanned=6)
display(file_count_by_repo1)

# query with DISTINCT commits without using DISTINCT, WITH, JOIN statements; 
# notice the SELECT is still working on an inner join

query2 = ("""
        -- Select all the columns we want in our joined table
        SELECT sc.repo_name, COUNT(sc.repo_name) AS number_of_commits
        FROM `bigquery-public-data.github_repos.sample_commits` as sc
        -- Table to merge into sample_files
        WHERE sc.repo_name IN 
        (SELECT sf.repo_name 
        FROM `bigquery-public-data.github_repos.sample_files` as sf
        WHERE sf.path LIKE ('%.py'))
        GROUP BY sc.repo_name
        ORDER BY number_of_commits DESC
        """)

file_count_by_repo2 = github.query_to_pandas_safe(query2, max_gb_scanned=6)
display(file_count_by_repo2)

# Removing duplicates using DISTINCT repo name from the sample file using WITH

query3 = ("""
        -- Select all the columns we want in our joined table
        WITH sample_file AS (
        SELECT DISTINCT repo_name  
        FROM `bigquery-public-data.github_repos.sample_files`
        WHERE path LIKE '%.py')
        SELECT sc.repo_name, COUNT(sc.commit) AS number_of_commits
        FROM `bigquery-public-data.github_repos.sample_commits` AS sc
        JOIN sample_file
        ON  sample_file.repo_name = sc.repo_name
        GROUP BY sc.repo_name
        ORDER BY number_of_commits DESC
        """)

file_count_by_repo3 = github.query_to_pandas_safe(query3, max_gb_scanned=6)
display(file_count_by_repo3)

