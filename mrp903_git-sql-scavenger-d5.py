import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import bq_helper

git = bq_helper.BigQueryHelper(active_project = 'bigquery-public-data', dataset_name = 'github_repos')
git.list_tables()
git.table_schema('sample_commits')
git.table_schema('sample_files')
query_1 = """SELECT DISTINCT sf.repo_name, COUNT(sc.commit) as Commits
             FROM `bigquery-public-data.github_repos.sample_files` AS sf
             INNER JOIN `bigquery-public-data.github_repos.sample_commits` AS sc
             ON sf.repo_name = sc.repo_name
             WHERE sf.path LIKE '%.py'
             GROUP BY sf.repo_name
             ORDER BY Commits DESC
          """
git.estimate_query_size(query_1)
repos_commits = git.query_to_pandas_safe(max_gb_scanned=7, query=query_1)
repos_commits
