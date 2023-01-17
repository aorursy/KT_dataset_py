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
# Python commits query
python_commits_q = """
        WITH python AS(
            SELECT repo_name
            FROM `bigquery-public-data.github_repos.sample_files`
            WHERE path LIKE '%.py'
        )
        SELECT commits.repo_name, COUNT(commit) AS commits_num
        FROM `bigquery-public-data.github_repos.sample_commits` AS commits
        JOIN python ON
        python.repo_name = commits.repo_name
        GROUP BY commits.repo_name
        ORDER BY commits_num DESC
        """
# Query size
github.estimate_query_size(python_commits_q)

# Python commits dataframe
python_commits = github.query_to_pandas_safe(python_commits_q, max_gb_scanned = 6)
            
python_commits
