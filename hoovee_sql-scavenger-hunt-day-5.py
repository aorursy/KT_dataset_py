# import package with helper functions 
import bq_helper

# create a helper object for this dataset
github = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                              dataset_name="github_repos")
query = """SELECT sf.repo_name AS repo, 
               COUNT(sc.commit) AS commits
           FROM `bigquery-public-data.github_repos.sample_commits` AS sc
           INNER JOIN `bigquery-public-data.github_repos.sample_files` as sf
               ON sf.repo_name = sc.repo_name
           WHERE sf.path LIKE '%.py'
           GROUP BY repo
           ORDER BY commits DESC
        """
print("Size of query:", github.estimate_query_size(query))
commits_in_repo = github.query_to_pandas_safe(query, max_gb_scanned=6)
# show amount of commits per repo that has python files in it
commits_in_repo