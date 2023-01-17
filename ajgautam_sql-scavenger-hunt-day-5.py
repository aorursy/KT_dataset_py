








# import package with helper functions 
import bq_helper

# create a helper object for this dataset
github = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                              dataset_name="github_repos")
#query to see how many commits have been made per repos written in python
#Using the 'sample_commits' and sample_files' tables
query = """ SELECT sf.repo_name AS repository_name, COUNT(DISTINCT sc.commit) AS commits
            FROM `bigquery-public-data.github_repos.sample_commits` as sc
            INNER JOIN `bigquery-public-data.github_repos.sample_files` as sf
                        ON sc.repo_name = sf.repo_name
            WHERE sf.path LIKE '%.py'
            GROUP BY repository_name
            ORDER BY commits DESC
            """

#Estimating the query size
print(github.estimate_query_size(query))
#max_GB scanned set to 6GB
commit_count_per_repo = github.query_to_pandas_safe(query, max_gb_scanned = 6)

#display the results
commit_count_per_repo
