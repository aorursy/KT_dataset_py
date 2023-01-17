import numpy as np
import pandas as pd
from google.cloud import bigquery
from bq_helper import BigQueryHelper
#For plots
import matplotlib.pyplot as plt
gh = BigQueryHelper('bigquery-public-data', 'github_repos')
gh_tables = gh.list_tables()
print("There are "+str(len(gh_tables))+" tables in the dataset")
print(gh_tables)
for x in range(0,len(gh_tables)):
    print("Table: "+gh_tables[x])
    a=gh.table_schema(gh_tables[x])
    for y in range(0,len(a)):
        print(a[y])
    print("\n\r") 
sql1="""SELECT COUNT(commit), repo_name
    FROM `bigquery-public-data.github_repos.sample_commits`
    WHERE repo_name IN (SELECT DISTINCT(repo_name)
    FROM `bigquery-public-data.github_repos.sample_files`
    WHERE path LIKE '%.py')
    GROUP BY repo_name
    """
gh.estimate_query_size(sql1)
repo = gh.query_to_pandas_safe(sql1,6)
repo.shape
repo_s=repo.sort_values(by=['f0_'],ascending=[False])
print(repo_s)
#Creating barplot
height = repo_s['f0_']
bars = repo_s['repo_name']
y_pos = np.arange(len(bars))
#Create bars and axis names
plt.bar(y_pos, height)
plt.xticks(y_pos, bars,rotation=-45)
#Show plot
plt.title("Number of commits pr repo_name in .py.")
plt.show()