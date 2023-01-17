# import package with helper functions 
import bq_helper

# create a helper object for this dataset
github = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                              dataset_name="github_repos")
# number of commits made in repos using Python (per *.py file path)
query = ("""
        WITH repos AS 
        (
            SELECT distinct sf.repo_name 
            FROM `bigquery-public-data.github_repos.sample_files` as sf
            WHERE path LIKE '%.py'
        )
        SELECT sc.repo_name, COUNT(sc.commit) as commit_count
        FROM `bigquery-public-data.github_repos.sample_commits` as sc
        INNER JOIN repos
            ON sc.repo_name = repos.repo_name
        GROUP BY sc.repo_name
        ORDER BY sc.repo_name
        """)

python_repos = github.query_to_pandas_safe(query, max_gb_scanned=6)
python_repos