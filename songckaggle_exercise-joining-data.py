# import package with helper functions 
import bq_helper

# create a helper object for this dataset
github = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                              dataset_name="github_repos")
# Your code here
query = """SELECT sf.repo_name, COUNT(C.commit) as number_of_commit
            FROM `bigquery-public-data.github_repos.sample_files` as sf
            INNER JOIN `bigquery-public-data.github_repos.sample_commits` as C 
                ON sf.repo_name = C.repo_name
            WHERE path LIKE '%.py'
            GROUP BY sf.repo_name
            ORDER BY number_of_commit DESC
        """
github.query_to_pandas_safe(query, max_gb_scanned=6)