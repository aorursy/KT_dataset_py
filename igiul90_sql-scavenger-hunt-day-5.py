# Your code goes here :)

# Import package with helper functions 
import bq_helper
import matplotlib.pyplot as plt

# Create a helper object for this dataset
github = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                              dataset_name="github_repos")

# Question1: How many commits have been made in repos written in the Python programming language?

question1 = """ SELECT COUNT(SF.path) AS Python_Files
                FROM `bigquery-public-data.github_repos.sample_files` AS SF
                INNER JOIN `bigquery-public-data.github_repos.sample_commits` AS SC
                ON SC.repo_name = SF.repo_name
            WHERE SF.path LIKE '%.py'
            """

# Estimation of query question1
print(github.estimate_query_size(question1))

# I use max_db_scanned = 21 to limit at 21 GB as Rachel suggest
commit = github.query_to_pandas_safe(question1, max_gb_scanned=21)

# Print Dataframe Size
print('Dataframe Size: {} Bytes'.format(int(commit.memory_usage(index=True, deep=True).sum())))

# Print Dataframe "transactions_per_day"
print(commit.head(20))