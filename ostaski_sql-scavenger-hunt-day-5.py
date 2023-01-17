# import package with helper functions 
import bq_helper

# create a helper object for this dataset
github = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                              dataset_name="github_repos")
# You can use two dashes (--) to add comments in SQL
query = ("""
        -- Select all the columns we want in our joined table
        SELECT L.license, COUNT(sf.path) AS number_of_files
        FROM `bigquery-public-data.github_repos.sample_files` as sf
        -- Table to merge into sample_files
        INNER JOIN `bigquery-public-data.github_repos.licenses` as L 
            ON sf.repo_name = L.repo_name -- what columns should we join on?
        GROUP BY L.license
        ORDER BY number_of_files DESC
        """)

file_count_by_license = github.query_to_pandas_safe(query, max_gb_scanned=6)
# print out all the returned results
print(file_count_by_license)
# importing the Big Query helper package
import bq_helper

# create a helper object for bigquery-public-data.github_repos
github = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                              dataset_name="github_repos")

# looking at the data
github.list_tables() 

# looking at the tables of interest
github.table_schema("sample_commits")
github.head("sample_commits")
github.table_schema("sample_files")
github.head("sample_files")

# Q1. How many commits (recorded in the "sample_commits" table) have been made in repos written in the Python programming language?
query = ("""
        -- Select the count of files in sample_files that are written in Python
        SELECT COUNT(sc.commit) AS number_of_commits, sf.repo_name
        FROM `bigquery-public-data.github_repos.sample_files` AS sf
        -- merge sample_commits into sample_files
        INNER JOIN `bigquery-public-data.github_repos.sample_commits` AS sc 
            ON sf.repo_name = sc.repo_name
        WHERE sf.path LIKE '%.py'
        GROUP BY sf.repo_name
        ORDER BY number_of_commits DESC
        """)

# estimate query size
github.estimate_query_size(query)
# 5.271098022349179, so bump the max_gb_scanned in query up to 6 gb

# run a "safe" query and store the resultset into a dataframe
python_commits = github.query_to_pandas_safe(query, max_gb_scanned=6)

# taking a look
print(python_commits)

# saving this in case we need it later
python_commits.to_csv("python_commits.csv")

