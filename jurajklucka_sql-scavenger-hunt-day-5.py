# import package with helper functions 
import bq_helper

# create a helper object for this dataset
github = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                              dataset_name="github_repos")
github.head('sample_files')
github.head('licenses')
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
# import package with helper functions 
import bq_helper

# create a helper object for this dataset
github = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                              dataset_name="github_repos")
github.head("sample_commits")
github.head("sample_files")
query = ("""
        SELECT sc.repo_name, COUNT(sc.commit) as number_of_commits
        FROM `bigquery-public-data.github_repos.sample_commits` as sc
        INNER JOIN `bigquery-public-data.github_repos.sample_files` as sf
        ON sc.repo_name=sf.repo_name
        WHERE path LIKE '%.py'
        GROUP BY sc.repo_name
        ORDER BY number_of_commits DESC
""")
repos_in_Python = github.query_to_pandas_safe(query,max_gb_scanned=9)
repos_in_Python


import matplotlib.pyplot as plt
import numpy as np

plt.barh(np.arange(len(repos_in_Python.repo_name)), repos_in_Python.number_of_commits, align='center', alpha=0.5)
plt.yticks(np.arange(len(repos_in_Python.repo_name)), repos_in_Python.repo_name)
plt.xlabel('Commits')
plt.title('Python commits by Repo')
plt.pie(repos_in_Python.number_of_commits, labels=repos_in_Python.repo_name, autopct='%1.1f%%', shadow=True)
plt.show()
