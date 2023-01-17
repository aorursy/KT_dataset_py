# import package with helper functions 
import bq_helper

# create a helper object for this dataset
github = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                              dataset_name="github_repos")
# print the first five rows of the "sample_files" table
github.head("sample_files")
# print the first five rows of the "sample_commits" table
github.head("sample_commits")
###QUESTION ###
#How many commits (recorded in the "sample_commits" table) 
#  have been made in repos written in the Python programming language? 
#(I'm looking for the number of commits per repo for all the repos written in Python.
#    * You'll want to JOIN the sample_files and sample_commits questions to answer this.
#    * **Hint:** You can figure out which files are written in Python by filtering results
#     from the "sample_files" table using `WHERE path LIKE '%.py'`. 
#     This will return results where the "path" column ends in the text ".py", which is 
#one way to identify which files have Python code.
query = ("""
        
        SELECT F.repo_name, COUNT(C.commit) AS number_of_commits
        FROM `bigquery-public-data.github_repos.sample_files` as F
        
        INNER JOIN `bigquery-public-data.github_repos.sample_commits` as C 
            ON F.repo_name = C.repo_name 
        where F.path like '%.py'
        GROUP BY 1
        ORDER BY 2 DESC
        """)

# Estimate query size
github.estimate_query_size(query)
# note that max_gb_scanned is set to 21, rather than 1
commits_per_repo = github.query_to_pandas_safe(query, max_gb_scanned=6)
print("number of commits per repo")
print(commits_per_repo)