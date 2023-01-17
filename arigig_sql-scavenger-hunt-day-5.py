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
# Your code goes here :)
# import package with helper functions 
import bq_helper

# create a helper object for this dataset
github = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                              dataset_name="github_repos")

# list all tables under this dataset "github_repos"
github.list_tables()

# following instructions print out the table structures of sample_commits and sample_files
github.table_schema("sample_commits")
github.table_schema("sample_files")

# print out few records out of both the tables
github.head("sample_commits")
github.head("sample_files")

# both tables are having a common column repo_name to be joined
# to get required result set
query = """SELECT sc.repo_name AS repo, COUNT(sc.commit) AS commit_count
             FROM `bigquery-public-data.github_repos.sample_commits` AS sc
       INNER JOIN `bigquery-public-data.github_repos.sample_files` AS sf
               ON sc.repo_name = sf.repo_name
            WHERE sf.path LIKE '%.py'
         GROUP BY sc.repo_name
         ORDER BY commit_count DESC
        """

# estimate query size
github.estimate_query_size(query)

# the query requires 5.27GB, run it in safe mode with 6GB
# store result in a dataframe
github_py_commits = github.query_to_pandas_safe(query,max_gb_scanned = 6)

# print out how many repo returned into the dataframe
github_py_commits.count()

# there are five repos written in python
# as the number of rows only 5, print out the entire table
print(github_py_commits)