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

#commented out to avoid running
#file_count_by_license = github.query_to_pandas_safe(query, max_gb_scanned=6)
# print out all the returned results
print(file_count_by_license)
#github.list_tables()
#github.table_schema('sample_files')
#github.table_schema('sample_commits')

query_1 = """SELECT count(sc.commit) AS number_of_commits_in_python,
                    sc.repo_name AS repository_name
                FROM `bigquery-public-data.github_repos.sample_commits` as sc,
                    `bigquery-public-data.github_repos.sample_files` as sf
                WHERE sc.repo_name = sf.repo_name AND sf.path LIKE '%.py'
                GROUP BY sc.repo_name
                ORDER BY number_of_commits_in_python DESC
                """
github.query_to_pandas_safe(query_1, max_gb_scanned = 6)