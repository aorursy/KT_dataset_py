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

github.table_schema(table_name="sample_commits")
github.head(table_name="sample_commits", selected_columns=["repo_name", "commit"])
github.head(table_name="sample_files", selected_columns=["repo_name", "path"])
query1 = """SELECT COUNT(sc.commit) AS number_of_commits, sf.repo_name
            FROM `bigquery-public-data.github_repos.sample_commits` as sc
            INNER JOIN `bigquery-public-data.github_repos.sample_files` as sf ON sc.repo_name = sf.repo_name
            WHERE sf.path LIKE '%.py'
            GROUP BY sf.repo_name
            ORDER BY number_of_commits DESC
         """

print(github.estimate_query_size(query=query1))

commits_per_python_repo_df = github.query_to_pandas_safe(query=query1, max_gb_scanned=6)
commits_per_python_repo_df.head()