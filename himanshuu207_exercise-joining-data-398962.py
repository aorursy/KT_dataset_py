# import package with helper functions 
import bq_helper

# create a helper object for this dataset
github = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                              dataset_name="github_repos")
query = """
        SELECT sf.path, COUNT(cm.commit) AS number_of_commits
        FROM `bigquery-public-data.github_repos.sample_files` as sf
        -- Table to merge into sample_files
        INNER JOIN `bigquery-public-data.github_repos.sample_commits` as cm
            ON sf.repo_name = cm.repo_name -- what columns should we join on?
        Group by sf.path
        Having sf.path LIKE '%py'
        ORDER BY number_of_commits DESC
        """

file_count_by_commits = github.query_to_pandas_safe(query, max_gb_scanned=50)
print (file_count_by_commits)