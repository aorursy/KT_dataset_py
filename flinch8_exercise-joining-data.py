# import package with helper functions 
import bq_helper

# create a helper object for this dataset
github = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                              dataset_name="github_repos")
query = """
        SELECT sf.repo_name, COUNT(sc.commit) as number_of_commits
        FROM `bigquery-public-data.github_repos.sample_files` as sf
        INNER JOIN `bigquery-public-data.github_repos.sample_commits` as sc
            ON sf.repo_name = sc.repo_name
        WHERE path LIKE '%.py'
        GROUP BY sf.repo_name
        ORDER BY number_of_commits
        """

python_commits = github.query_to_pandas_safe(query, max_gb_scanned=8)

print(python_commits.head())