# import package with helper functions 
import bq_helper

# create a helper object for this dataset
github = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                              dataset_name="github_repos")
query = """SELECT sf.repo_name, COUNT(c.commit) as commit_amount
            FROM `bigquery-public-data.github_repos.sample_files` as sf
            INNER JOIN `bigquery-public-data.github_repos.sample_commits` as c
            ON  sf.repo_name = c.repo_name
            WHERE path LIKE '%.py'
            GROUP BY sf.repo_name
            ORDER BY commit_amount DESC"""

github.query_to_pandas(query)