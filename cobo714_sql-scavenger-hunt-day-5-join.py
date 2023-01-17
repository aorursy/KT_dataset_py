import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import bq_helper
github = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                  dataset_name="github_repos")
github.list_tables()
github.table_schema("sample_files")
github.table_schema("licenses")
query = """SELECT L.license AS License,
                  COUNT(sf.id) AS NumFiles
           FROM `bigquery-public-data.github_repos.sample_files` AS sf
           INNER JOIN `bigquery-public-data.github_repos.licenses` AS L
                   ON sf.repo_name = L.repo_name
           GROUP BY License
           ORDER BY NumFiles DESC
        """
file_count_by_license = github.query_to_pandas_safe(query, max_gb_scanned=6)
file_count_by_license
github.table_schema("sample_commits")
github.table_schema("sample_files")
test = """SELECT DISTINCT repo_name
          FROM `bigquery-public-data.github_repos.sample_files`
       """
uniq_repos = github.query_to_pandas_safe(test, max_gb_scanned=2)
scav1 = """WITH python_repos AS
           (
               SELECT DISTINCT repo_name
               FROM `bigquery-public-data.github_repos.sample_files`
               WHERE path LIKE '%.py'
           )
           
           SELECT sC.repo_name AS Repos,
                  COUNT(sC.commit) AS NumCommits
           FROM `bigquery-public-data.github_repos.sample_commits` AS sC
           INNER JOIN python_repos
                   ON sc.repo_name = python_repos.repo_name
           GROUP BY Repos
           ORDER BY NumCommits DESC
        """
python_commits = github.query_to_pandas_safe(scav1, max_gb_scanned=6)
python_commits