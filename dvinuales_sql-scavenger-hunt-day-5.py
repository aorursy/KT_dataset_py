# Your code goes here :)

# import package with helper functions 
import bq_helper

# create a helper object for this dataset
github = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                              dataset_name="github_repos")

# Added DISTINCT to avoid repeated commit
query = ("""        
        SELECT sf.repo_name as repo, COUNT(DISTINCT sc.commit) AS commits
        FROM `bigquery-public-data.github_repos.sample_commits` as sc
        INNER JOIN `bigquery-public-data.github_repos.sample_files` as sf
            ON sf.repo_name = sc.repo_name
        WHERE sf.path LIKE '%.py'
        GROUP BY repo
        ORDER BY commits DESC
        """)

python_commits = github.query_to_pandas_safe(query, max_gb_scanned=6)

# print out all the returned results
print(python_commits)
import matplotlib.pyplot as plt
import numpy as np

plt.barh(np.arange(len(python_commits.repo)), python_commits.commits, align='center', alpha=0.5)
plt.yticks(np.arange(len(python_commits.repo)), python_commits.repo)
plt.xlabel('Commits')
plt.title('Python commits by Repo')
plt.pie(python_commits.commits, labels=python_commits.repo, autopct='%1.1f%%', shadow=True)
plt.show()