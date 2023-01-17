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
#idea: return repository (those in Python) and associated number of commits (as a count)
query_one = """--make a temp table of repo name that has py path; based on others' feedback
               --need to use distinct repo name in sf to avoid multiple counts of the same repo
                WITH repo_py AS
            (
                SELECT DISTINCT repo_name
                FROM `bigquery-public-data.github_repos.sample_files`
                WHERE path LIKE '%.py'
            )
--select repo name and counts of commits from sample commit
SELECT sc.repo_name, COUNT(sc.commit)AS number_of_commits
FROM `bigquery-public-data.github_repos.sample_commits` AS sc

--join temp table repo_py with sc 
INNER JOIN repo_py 
    ON repo_py.repo_name = sc.repo_name
    
--in the joined table, group by repo name and order the counts    
GROUP BY repo_name
ORDER BY number_of_commits DESC
"""

commit_count = github.query_to_pandas_safe(query_one, max_gb_scanned=7)

# print out all the returned results
print(commit_count)
