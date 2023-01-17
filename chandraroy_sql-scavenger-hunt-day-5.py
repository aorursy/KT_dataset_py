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
# Your code goes here :)
import bq_helper
commits = bq_helper.BigQueryHelper(active_project= "bigquery-public-data", 
                                   dataset_name = 'github_repos' )
commits.list_tables()
commits.table_schema('sample_commits')
commits.table_schema('sample_files')
commits.head("sample_files")
commits.head("sample_commits")
query = ("""
        -- list all the columns names
        SELECT sp.repo_name, COUNT(sc.commit)
        FROM `bigquery-public-data.github_repos.sample_commits` as sc
        INNER JOIN `bigquery-public-data.github_repos.sample_files` as sp 
        ON sc.repo_name = sp.repo_name
        WHERE path LIKE '%.py'
        GROUP BY repo_name
        """      
        )
commits.estimate_query_size(query)
commits.query_to_pandas_safe(query, max_gb_scanned=6)

number_commits = commits.query_to_pandas_safe(query, max_gb_scanned=6)
import matplotlib as plt
number_commits.plot()
