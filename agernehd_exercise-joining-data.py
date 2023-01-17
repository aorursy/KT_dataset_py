# import package with helper functions 
import bq_helper

# create a helper object for this dataset
github = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                              dataset_name="github_repos")
query = """
        -- select sample files and commits
        SELECT
            L.license,
            COUNT(sf.path) AS number_of_python_commits  
        FROM `bigquery-public-data.github_repos.sample_files` as sf
        INNER JOIN `bigquery-public-data.github_repos.sample_commits` as sc
         -- column to joine on
            ON sf.repo_name = sc.repo_name 
        INNER JOIN `bigquery-public-data.github_repos.licenses` as L            
            on sf.repo_name = L.repo_name
        WHERE sf.path LIKE '%.py'
        GROUP BY L.license
        ORDER BY number_of_python_commits

"""
number_of_python_commits = github.query_to_pandas_safe(query, max_gb_scanned=6)
print("Number of Python Commits by License Type: ", number_of_python_commits.values)
# Using With .. AS to tidy the query
query_2 = """
         -- create CTE for python commits
          WITH py_commits AS (
              SELECT
                sc.repo_name,
                sf.path AS py_commits  
              FROM `bigquery-public-data.github_repos.sample_files` as sf
              INNER JOIN `bigquery-public-data.github_repos.sample_commits` as sc
                 -- column to joine on
                    ON sf.repo_name = sc.repo_name 
              WHERE sf.path LIKE '%.py'              
          )  
        
        -- merge license and num_py_commits
        SELECT L.license,
                 count(py.py_commits) AS number_of_python_commits
        FROM py_commits as py
        INNER JOIN `bigquery-public-data.github_repos.licenses` as L            
            on py.repo_name = L.repo_name        
        GROUP BY L.license
        ORDER BY number_of_python_commits

"""
number_of_python_commits_2 = github.query_to_pandas_safe(query, max_gb_scanned=6)

print(number_of_python_commits_2)