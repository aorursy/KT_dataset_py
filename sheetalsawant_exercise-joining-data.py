# import package with helper functions 
import bq_helper

# create a helper object for this dataset
github = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                              dataset_name="github_repos")
github.head("sample_files")
github.head("sample_commits")
# Your code here
query = """SELECT sf.repo_name AS repo,COUNT(sc.commit) AS no_of_commits
           FROM `bigquery-public-data.github_repos.sample_files` AS sf
           INNER JOIN `bigquery-public-data.github_repos.sample_commits` AS sc
           ON sf.repo_name=sc.repo_name
           WHERE sf.path LIKE '%.py'
           GROUP BY sf.repo_name
           ORDER BY no_of_commits
        """
github.query_to_pandas_safe(query, max_gb_scanned=8)

