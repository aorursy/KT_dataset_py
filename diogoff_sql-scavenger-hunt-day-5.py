import bq_helper
github = bq_helper.BigQueryHelper(active_project='bigquery-public-data',
                                  dataset_name='github_repos')
github.list_tables()
github.head('sample_files')
github.head('licenses')
query = '''SELECT l.license, COUNT(sf.path) AS num_files
           FROM `bigquery-public-data.github_repos.sample_files` as sf,
                `bigquery-public-data.github_repos.licenses` as l
           WHERE sf.repo_name = l.repo_name
           GROUP BY l.license
           ORDER BY num_files DESC'''
github.estimate_query_size(query)
num_files_by_license = github.query_to_pandas_safe(query, max_gb_scanned=6)
num_files_by_license
num_files_by_license.plot(x='license', y='num_files', kind='bar')
github.head('sample_commits')
query = '''SELECT sc.repo_name, COUNT(sc.commit) AS num_commits
           FROM `bigquery-public-data.github_repos.sample_commits` as sc,
                `bigquery-public-data.github_repos.sample_files` as sf
           WHERE (sc.repo_name = sf.repo_name) AND (sf.path LIKE '%.py')
           GROUP BY sc.repo_name
           ORDER BY num_commits DESC'''
github.estimate_query_size(query)
num_commits_by_repo = github.query_to_pandas_safe(query, max_gb_scanned=6)
num_commits_by_repo.head()
num_commits_by_repo.plot(x='repo_name', y='num_commits', kind='bar')