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

github.head("sample_commits")
github.head("sample_files")
# Your code goes here :)
'''
*  How many commits (recorded in the "sample_commits" table) have been made in repos written 
in the Python programming language?
    * You'll want to JOIN the sample_files and sample_commits questions to answer this.
    * **Hint:** You can figure out which files are written in Python by filtering results 
    from the "sample_files" table using `WHERE path LIKE '%.py'`. 
    This will return results where the "path" column ends in the text ".py", 
    which is one way to identify which files have Python code.
'''
# You can use two dashes (--) to add comments in SQL
query = ("""
        SELECT sf.repo_name, COUNT(sf.path) AS py_count
        FROM `bigquery-public-data.github_repos.sample_files` as sf
        WHERE ENDS_WITH(sf.path, '.py')
        GROUP BY sf.repo_name
        ORDER BY py_count DESC
        """)

py_count_by_repos = github.query_to_pandas_safe(query, max_gb_scanned=6)

# print out all the returned results
print(py_count_by_repos)
# Your code goes here :)
'''
*  How many commits (recorded in the "sample_commits" table) have been made in repos written 
in the Python programming language?
    * You'll want to JOIN the sample_files and sample_commits questions to answer this.
    * **Hint:** You can figure out which files are written in Python by filtering results 
    from the "sample_files" table using `WHERE path LIKE '%.py'`. 
    This will return results where the "path" column ends in the text ".py", 
    which is one way to identify which files have Python code.
'''
# You can use two dashes (--) to add comments in SQL
query = ("""
        SELECT sf.repo_name, COUNT(sc.commit) AS number_of_commits
        FROM `bigquery-public-data.github_repos.sample_files` as sf
        INNER JOIN `bigquery-public-data.github_repos.sample_commits` as sc 
            ON sf.repo_name = sc.repo_name AND ENDS_WITH(sf.path, '.py')
        GROUP BY sf.repo_name
        ORDER BY number_of_commits DESC
        """)

commit_count_by_repos = github.query_to_pandas_safe(query, max_gb_scanned=6)

# print out all the returned results
print(commit_count_by_repos)