import bq_helper

# create a helper object for this dataset
github = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                              dataset_name="github_repos")
#print a list of all the tables in the github_repos dataset
github.list_tables()
# print information on all the columns in the "sample_files" table
# in the github dataset
github.table_schema("sample_files")
github.table_schema("sample_commits")
query = ("""
        -- Select all the columns we want in our joined table
        SELECT P.repo_name, COUNT(sc.commit) AS number_of_commits
        FROM `bigquery-public-data.github_repos.sample_commits` as sc
        -- Table to merge into sample_files
        INNER JOIN `bigquery-public-data.github_repos.sample_files` as P
            ON sc.repo_name = P.repo_name -- what columns should we join on?
        WHERE p.repo_name LIKE '%.py'
        GROUP by P.repo_name
        ORDER by number_of_commits
        """)
commit_count = github.query_to_pandas_safe(query, max_gb_scanned = 10)
commit_count