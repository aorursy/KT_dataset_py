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
github.list_tables()
github.head('sample_commits')
github.head('sample_files')
my_query = """SELECT f.repo_name, COUNT(c.commit) AS num
                FROM `bigquery-public-data.github_repos.sample_files` AS f
                INNER JOIN `bigquery-public-data.github_repos.sample_commits` AS c
                ON f.repo_name = c.repo_name
                WHERE f.path LIKE '%.py'
                GROUP BY f.repo_name
                ORDER BY num DESC"""

commits_per_python_repo = github.query_to_pandas_safe(my_query, max_gb_scanned=9)
commits_per_python_repo.head()
# according to a Joseph Corliss suggestion
my_query_dist = """
                WITH python_repos AS (
                    SELECT DISTINCT repo_name
                    FROM `bigquery-public-data.github_repos.sample_files`
                    WHERE path LIKE '%.py')
                SELECT c.repo_name, COUNT(c.commit) AS num
                FROM `bigquery-public-data.github_repos.sample_commits` AS c
                JOIN python_repos
                ON python_repos.repo_name = c.repo_name
                GROUP BY c.repo_name
                ORDER BY num DESC"""

commits_per_python_repo = github.query_to_pandas_safe(my_query_dist, max_gb_scanned=9)
commits_per_python_repo.head()