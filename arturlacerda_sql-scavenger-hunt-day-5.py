# import package with helper functions 
import bq_helper

# create a helper object for this dataset
github = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                  dataset_name="github_repos")
github.list_tables()
github.head("sample_commits")
github.head("sample_files")
query = """ WITH Python AS
            (
                SELECT DISTINCT repo_name
                FROM `bigquery-public-data.github_repos.sample_files`
                WHERE path LIKE '%.py'
            )
            SELECT Python.repo_name, COUNT(DISTINCT Commits.commit) AS num_commits
            FROM `bigquery-public-data.github_repos.sample_commits` as Commits
            INNER JOIN python
                ON Python.repo_name = Commits.repo_name
            GROUP BY Python.repo_name
            ORDER BY num_commits DESC
        """
# This solution also works, but is less elegant:
# query = """ SELECT Files.repo_name AS repo_name, COUNT(DISTINCT Commits.commit) AS num_commits
#             FROM `bigquery-public-data.github_repos.sample_files` as Files
#             INNER JOIN `bigquery-public-data.github_repos.sample_commits` as Commits
#                 ON Files.repo_name = Commits.repo_name
#             WHERE Files.path LIKE '%.py'
#             GROUP BY Files.repo_name
#             ORDER BY num_commits DESC
#         """
github.estimate_query_size(query)
python_commits = github.query_to_pandas_safe(query, max_gb_scanned=6)
python_commits