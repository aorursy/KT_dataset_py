# import package with helper functions 
import bq_helper

# create a helper object for this dataset
github = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                              dataset_name="github_repos")
github.list_tables()
github.head('files')
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
github.estimate_query_size(query)
file_count_by_license = github.query_to_pandas_safe(query, max_gb_scanned=6)
# print out all the returned results
print(file_count_by_license)
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
plt.style.use('ggplot')
sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5})
fig, bars = plt.subplots(figsize=(12, 9))
bars = sns.barplot(x="license", y="number_of_files", data=file_count_by_license)
bars.set_xticklabels(bars.get_xticklabels(), rotation=90)
plt.title("Licenses Used on GitHub")
plt.show(bars)
plt.bar(file_count_by_license.license, file_count_by_license.number_of_files, align='center', alpha=0.5)
plt.xticks(file_count_by_license.license,  ha='right', rotation=45)
plt.ylabel('Number of files')
plt.title('Licenses Used in Github')
plt.show()
# 1. Examine the table 
github.head('sample_commits')
github.head('sample_files')
comments_per_repo_query = """
                    SELECT sf.path,COUNT(sc.commit) AS number_of_commits
                    FROM `bigquery-public-data.github_repos.sample_files` as sf
                    INNER JOIN `bigquery-public-data.github_repos.sample_commits` as sc 
                        ON sf.repo_name = sc.repo_name
                    WHERE sf.path LIKE '%.py'
                    GROUP BY path
                    """
github.estimate_query_size(comments_per_repo_query)
comments_per_repo = github.query_to_pandas_safe(comments_per_repo_query, max_gb_scanned=6)
print(comments_per_repo)
#Total number of repo written in python are 
comments_per_repo.number_of_commits.sum()