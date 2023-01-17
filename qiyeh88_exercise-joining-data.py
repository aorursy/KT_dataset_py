# import package with helper functions 
import bq_helper

# create a helper object for this dataset
github = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                              dataset_name="github_repos")
github.list_tables()
github.table_schema("sample_files")
github.table_schema('sample_commits')
query = """WITH python_files AS
                (
                    SELECT DISTINCT repo_name --There is a need for DISTINCT over here! If not there will be duplicates copy!
                    FROM `bigquery-public-data.github_repos.sample_files`
                    WHERE path LIKE '%.py'
                )
            SELECT python_files.repo_name,
                COUNT(com.commit) AS num_commits
            FROM `bigquery-public-data.github_repos.sample_commits` AS com
            INNER JOIN python_files ON com.repo_name = python_files.repo_name
            GROUP BY python_files.repo_name
        """
github.estimate_query_size(query)
output_commits_python_repo = github.query_to_pandas_safe(query,max_gb_scanned= 6)
output_commits_python_repo
query = """WITH python_files AS
                (
                    SELECT repo_name --There is a need for DISTINCT over here! If not there will be duplicates copy!
                    FROM `bigquery-public-data.github_repos.sample_files`
                    WHERE path LIKE '%.py'
                )
            SELECT python_files.repo_name,
                COUNT(com.commit) AS num_commits
            FROM `bigquery-public-data.github_repos.sample_commits` AS com
            INNER JOIN python_files ON com.repo_name = python_files.repo_name
            GROUP BY python_files.repo_name
        """
github.estimate_query_size(query)
output_commits_python_repo2 = github.query_to_pandas_safe(query,max_gb_scanned= 6)
output_commits_python_repo2