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
query1 = """
        with repodata as
        (
        SELECT sc.repo_name as reponame, sc.commit as commits, sf.path as path 
        FROM `bigquery-public-data.github_repos.sample_files` as sf inner join `bigquery-public-data.github_repos.sample_commits` as sc on sf.repo_name=sc.repo_name 
        WHERE sf.path like '%.py'
     
        )
        
         SELECT reponame as Repository_Name, count(commits) as No_of_Commits
         FROM repodata 
         GROUP BY reponame
         ORDER BY No_of_Commits DESC
        
        
        
        
        
        """

repo_name = github.query_to_pandas_safe(query1, max_gb_scanned=20)



print(repo_name.head())

import matplotlib.pyplot as plt
plt.barh(repo_name.Repository_Name,repo_name.No_of_Commits,log=True)

#Here log=True will set the axis on logarithmic scale and hence wewill be able to view all the data
#as the scale for the data is not same