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
query9 =  """
            WITH python_repos AS (
                SELECT repo_name
                FROM `bigquery-public-data.github_repos.sample_files`
                WHERE path LIKE '%.py')
                SELECT commits.repo_name, COUNT(commit) AS num_commits
                FROM `bigquery-public-data.github_repos.sample_commits` AS commits
                JOIN python_repos
                ON  python_repos.repo_name = commits.repo_name
                GROUP BY commits.repo_name
                ORDER BY num_commits DESC
        """ 
    
commit_count_by_name = github.query_to_pandas(query9)

print(commit_count_by_name)

#OU
query = ( """
        -- Select all the columns we want in our joined table
        SELECT sf.repo_name, COUNT(sf.commit) AS num_commits
        FROM `bigquery-public-data.github_repos.sample_commits` as sf
        -- Table to merge into sample_files
        INNER JOIN `bigquery-public-data.github_repos.sample_files` as L 
        ON sf.repo_name = L.repo_name 
        WHERE L.path LIKE '%.py'-- what columns should we join on?
        GROUP BY sf.repo_name
        ORDER BY num_commits DESC
        """ )

commit_count_by_name = github.query_to_pandas_safe(query, max_gb_scanned=6)

print(commit_count_by_name)