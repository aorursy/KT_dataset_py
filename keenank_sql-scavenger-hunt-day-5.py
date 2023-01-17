import bq_helper

github = bq_helper.BigQueryHelper(active_project="bigquery-public-data", dataset_name="github_repos")
github.head("sample_files")
github.head("sample_commits")
# Old query that has incorrect solution. Not distinct list of Repos so the number of commits
# is multiplied by the number of python files within that repo
#query = """ SELECT COUNT(sc.commit) AS number_commits_python, sf.repo_name AS repo_name
#            FROM `bigquery-public-data.github_repos.sample_commits` AS sc
#            INNER JOIN `bigquery-public-data.github_repos.sample_files` AS sf
#                ON sc.repo_name = sf.repo_name
#            WHERE sf.path like '%.py'
#            GROUP BY repo_name
#            ORDER BY number_commits_python DESC
#        """

query = """ WITH repo_py AS
            (
                SELECT DISTINCT repo_name
                FROM `bigquery-public-data.github_repos.sample_files`
                WHERE path LIKE '%.py'
            )
            SELECT COUNT(sc.commit) AS number_of_commits, sc.repo_name
            FROM `bigquery-public-data.github_repos.sample_commits` AS sc
            INNER JOIN repo_py
                ON repo_py.repo_name = sc.repo_name
            GROUP BY repo_name
            ORDER BY number_of_commits DESC
        """

commits_repo_py = github.query_to_pandas_safe(query, max_gb_scanned=6)
print(commits_repo_py)