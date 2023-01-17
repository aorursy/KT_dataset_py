# Import package with helper functions 
%matplotlib inline
import bq_helper

# Create a helper object for this dataset
github = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                              dataset_name="github_repos")
# Display the structure of the GitHub Repos dataset
github.list_tables()
# Display first 5 rows of 'sample_commits' table
github.head('sample_commits')
# Display first 5 rows of 'sample_files' table
github.head('sample_files')
query_1 = ("""SELECT sf.repo_name as Repository, COUNT(sc.commit) AS NumOfCommits
              FROM `bigquery-public-data.github_repos.sample_commits` AS sc
              INNER JOIN `bigquery-public-data.github_repos.sample_files` AS sf
                  ON sf.repo_name  = sc.repo_name
              WHERE sf.path LIKE '%.py'
              GROUP BY sf.repo_name
              ORDER BY NumOfCommits DESC
           """)

python_commits = github.query_to_pandas_safe(query_1, max_gb_scanned=6)
# Display first 5 rows of our result table
python_commits.head()
# Import matplotlib.pyplot and seaborn plotting libraries
import matplotlib.pyplot as plt
import seaborn as sns


# Set plot parameters
fig, ax = plt.subplots()
fig.set_size_inches(12.7, 8.27)
g = sns.barplot(x = 'Repository', y = 'NumOfCommits', data = python_commits, 
                palette = sns.cubehelix_palette(8, start=.5, rot=-.75), log = True)
ax.set(xlabel='Repository', ylabel='Number of Python commits')
ax.set_title('Python commits')

# Display number of commits above each bar
for index, value in python_commits.NumOfCommits.iteritems():
    ax.text(index,value, value, horizontalalignment = 'center', verticalalignment = 'bottom')