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
pythonCommitsQuery = """WITH pythonRepos AS
                        (
                        SELECT DISTINCT repo_name
                        FROM `bigquery-public-data.github_repos.sample_files`
                        WHERE path LIKE '%.py'
                        )
                        SELECT sc.repo_name, COUNT(*) as commits
                        FROM `bigquery-public-data.github_repos.sample_commits` as sc
                        INNER JOIN pythonRepos ON pythonRepos.repo_name = sc.repo_name
                        GROUP BY sc.repo_name
                        ORDER BY commits DESC
                    """

pythonCommits = github.query_to_pandas_safe(pythonCommitsQuery, max_gb_scanned = 6)
pythonCommits.head()