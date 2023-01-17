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
# Answer
# First we'll have a look at tables schemas to identify common key to join on
github.table_schema('sample_files')
github.table_schema('sample_commits')
# As we can see, the possible join is `repo_name`
query = """select files.dist_repo Repo_name, 
       count(*) No_of_commits
       from (select distinct(repo_name) dist_repo
             from `bigquery-public-data.github_repos.sample_files`
             where path like '%.py') files
        inner join `bigquery-public-data.github_repos.sample_commits` commits
            on files.dist_repo = commits.repo_name
        group by files.dist_repo
        order by No_of_commits desc;
        """
commits_data = github.query_to_pandas_safe(query, max_gb_scanned = 6)
commits_data
# Simple plot to visualize results
from matplotlib import pyplot as plt
plt.barh(commits_data.Repo_name, commits_data.No_of_commits)
plt.title("Github repos (containing Python) files\nwith most commits")
plt.xticks(fontsize=10,rotation = 70);