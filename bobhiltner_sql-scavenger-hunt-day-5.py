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
# How mnay commits per repo.  
# Wrong version (see below)--includes only python commits.
query = """
    select files.repo_name as repo, 
    count(*) as commit_count
    from `bigquery-public-data.github_repos.sample_commits` as commits
    INNER JOIN
    `bigquery-public-data.github_repos.sample_files` as files
    ON files.repo_name = commits.repo_name
    where files.path LIKE '%.py'
    GROUP BY files.repo_name
"""
python_commits_by_repo = github.query_to_pandas_safe(query, max_gb_scanned=6)
python_commits_by_repo.head()
# Careful, joining on just python files misses repo commits for other files .
# How mnay commits per repo.  
query = """
    with repo as (
        SELECT DISTINCT files.repo_name
        from 
            `bigquery-public-data.github_repos.sample_files` as files
        INNER JOIN
        `bigquery-public-data.github_repos.sample_commits` as commits
        ON 
            files.repo_name = commits.repo_name
        where 
            files.path LIKE '%.py'
        )
    select repo.repo_name, 
    count(*) as commit_count
    from 
        repo
    INNER JOIN
        `bigquery-public-data.github_repos.sample_commits` as commits
    ON 
        repo.repo_name = commits.repo_name
    GROUP BY repo.repo_name
"""
python_commits_by_repo = github.query_to_pandas_safe(query, max_gb_scanned=6)
python_commits_by_repo.head()