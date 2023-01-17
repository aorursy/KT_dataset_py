# Import package with helper functions
import bq_helper

# create a helper object for this dataset
github = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                                 dataset_name="github")

# our first query, How many commits (recorded in the "sample_commits" tables) have been made in
# repos written in the Python programming language

query = (
        """
        SELECT COUNT(commit) AS pythonCommits
        FROM `bigquery-public-data.github_repos.sample_commits` AS sc
            INNER JOIN `bigquery-public-data.github_repos.sample_files` AS sf
                ON sc.repo_name = sf.repo_name
        WHERE sf.path LIKE '%.py'
        """
        )

# Check the size
github.estimate_query_size(query)

# Perform our query
countIsPython= github.query_to_pandas_safe(query, max_gb_scanned=6)

# How many?
print(countIsPython)