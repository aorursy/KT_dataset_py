# setup
import bq_helper

github = bq_helper.BigQueryHelper(active_project='bigquery-public-data',
                                 dataset_name='github_repos')
# explore tables
github.list_tables()
# total Python commits
query1 = '''SELECT COUNT(*) AS python_commits
        FROM `bigquery-public-data.github_repos.sample_files` AS f
        INNER JOIN `bigquery-public-data.github_repos.sample_commits` AS c ON c.repo_name = f.repo_name
        WHERE path LIKE '%.py'
        '''
github.query_to_pandas_safe(query1, max_gb_scanned=10)
# breakdown of python commits based on repo
query2 = '''SELECT c.repo_name, COUNT(*) AS commits
        FROM `bigquery-public-data.github_repos.sample_files` AS f
        INNER JOIN `bigquery-public-data.github_repos.sample_commits` AS c ON c.repo_name = f.repo_name
        WHERE path LIKE '%.py'
        GROUP BY 1
        ORDER BY 2 DESC
        '''
result2= github.query_to_pandas_safe(query2, max_gb_scanned=10)
result2
