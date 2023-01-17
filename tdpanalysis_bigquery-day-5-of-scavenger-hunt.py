import bq_helper as bq
github_repos = bq.BigQueryHelper(active_project = "bigquery-public-data",
                                      dataset_name = "github_repos")
query = """ SELECT F.repo_name,
                   COUNT(C.commit) AS commits
            FROM `bigquery-public-data.github_repos.sample_files` AS F
            INNER JOIN `bigquery-public-data.github_repos.sample_commits` AS C
                ON F.repo_name = C.repo_name
            WHERE F.path LIKE '%.py%'
            GROUP BY F.repo_name
            ORDER BY commits DESC
"""
github_repos.estimate_query_size(query)
query_results = github_repos.query_to_pandas_safe(query, max_gb_scanned=6)
query_results.head(10)
query2 = """SELECT repo_name,
                   COUNT(path) py_files
            FROM `bigquery-public-data.github_repos.sample_files`
            WHERE path LIKE '%.py' 
                      AND
                  repo_name IN ('Microsoft/vscode','facebook/react','apple/swift',
                                'tensorflow/tensorflow','torvalds/linux')
            GROUP BY repo_name
            ORDER BY py_files DESC
"""
github_repos.estimate_query_size(query2)
query2_results = github_repos.query_to_pandas_safe(query2,max_gb_scanned=6)
query2_results.head(10)
query3 = """WITH py_repos AS (
            SELECT DISTINCT(repo_name)
            FROM `bigquery-public-data.github_repos.sample_files`
            WHERE path LIKE '%.py')
                SELECT C.repo_name,
                       COUNT(commit) AS commits
                FROM `bigquery-public-data.github_repos.sample_commits` AS C
                INNER JOIN py_repos
                    ON C.repo_name = py_repos.repo_name
                GROUP BY C.repo_name
                ORDER BY commits DESC
"""
github_repos.estimate_query_size(query3)
commits_per_repo = github_repos.query_to_pandas_safe(query3, max_gb_scanned=6)
commits_per_repo.head(10)
commits_per_repo.sort_values('commits').plot.barh('repo_name','commits')