# import package with helper functions 
import bq_helper

# create a helper object for this dataset
github = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                              dataset_name="github_repos")
github.list_tables()
github.head("sample_commits")
# import package with helper functions 
import bq_helper

# create a helper object for this dataset
github = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                              dataset_name="github_repos")
github.list_tables()
github.head("sample_files")
query = """
            SELECT COUNT(c.commit) AS Number_of_Commits, r.repo_name as Repos
            FROM `bigquery-public-data.github_repos.sample_commits` AS c
            INNER JOIN `bigquery-public-data.github_repos.sample_files` AS r
                ON c.repo_name = r.repo_name
            WHERE r.path LIKE '%.py'
            GROUP BY Repos
            ORDER BY Number_of_Commits DESC
        """
python_commits = github.query_to_pandas_safe(query, max_gb_scanned=6)
print(python_commits)