import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from subprocess import check_output
import bq_helper 
GitHub_Repos = bq_helper.BigQueryHelper(active_project= "bigquery-public-data", 
                                       dataset_name = "github_repos")
# print a list of all the tables in the dataset
GitHub_Repos.list_tables()
GitHub_Repos.head("licenses", num_rows=10)
query = '''        
           SELECT L.license, COUNT(sf.path) AS number_of_files -- Select all the columns we want in our joined table
           FROM `bigquery-public-data.github_repos.sample_files` as sf -- Table to merge into sample_files
           INNER JOIN `bigquery-public-data.github_repos.licenses` as L 
                       ON sf.repo_name = L.repo_name -- what columns should we join on?
           GROUP BY L.license
           ORDER BY number_of_files DESC
           '''
GitHub_Repos.estimate_query_size(query)
df = GitHub_Repos.query_to_pandas_safe(query, max_gb_scanned=6)
df.head(10)
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
plt.style.use('ggplot')
sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5})
fig, bars = plt.subplots(figsize=(12, 9))
bars = sns.barplot(x="license", y="number_of_files", data=df)
bars.set_xticklabels(bars.get_xticklabels(), rotation=90)
plt.title("Licenses Used on GitHub")
plt.show(bars)
GitHub_Repos.head("sample_files")
GitHub_Repos.head("sample_commits")
query = '''        
           SELECT COUNT(commit) AS n_of_commits_per_repo 
           FROM `bigquery-public-data.github_repos.sample_commits` as SC 
           INNER JOIN `bigquery-public-data.github_repos.sample_files` as SF 
                       ON SC.repo_name = SF.repo_name 
                       WHERE SF.path LIKE '%.py'
           '''
GitHub_Repos.estimate_query_size(query)
df = GitHub_Repos.query_to_pandas_safe(query, max_gb_scanned=6)
df
