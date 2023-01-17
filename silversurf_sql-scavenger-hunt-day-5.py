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
q_python_commits = '''
    WITH python_files_repo AS
    (
    SELECT
        DISTINCT repo_name AS python_reponame
    FROM `bigquery-public-data.github_repos.sample_files`
    WHERE path LIKE '%.py'
    )    
    SELECT
        repo_name,
        COUNT(*) AS commit_num
    FROM `bigquery-public-data.github_repos.sample_commits` samp_comm
    INNER JOIN python_files_repo py_repo ON py_repo.python_reponame=samp_comm.repo_name
    GROUP BY repo_name
    ORDER BY commit_num DESC
'''
github.estimate_query_size(q_python_commits)
commits_per_repo_df = github.query_to_pandas(q_python_commits)
commits_per_repo_df
q_test0 = '''
    WITH samp_files_repos AS
    (
    SELECT DISTINCT repo_name
    FROM `bigquery-public-data.github_repos.sample_files`
    )
    
    SELECT DISTINCT comm.repo_name
    FROM `bigquery-public-data.github_repos.sample_commits` comm
    INNER JOIN samp_files_repos f_rep ON f_rep.repo_name=comm.repo_name
'''
github.estimate_query_size(q_test0)
github.query_to_pandas(q_test0)
q_commits_per_repos_big = '''
    
    WITH py_repos AS
    (
    SELECT DISTINCT repo_name
    FROM `bigquery-public-data.github_repos.files`
    ),
    
     repo_names AS
    (
    SELECT
        repo_name AS repo_name
    FROM `bigquery-public-data.github_repos.commits`
    ),
    
    commits_per_reponame AS
    (SELECT
        flatten_repo_name,
        COUNT(*) commits_number
    FROM
        repo_names
    CROSS JOIN
        UNNEST(repo_names.repo_name) AS flatten_repo_name
    GROUP BY flatten_repo_name
    )
    
    SELECT
        flatten_repo_name,
        commits_number
    FROM
        commits_per_reponame com_per_repo
    INNER JOIN py_repos py_rep ON py_rep.repo_name=com_per_repo.flatten_repo_name
    ORDER BY commits_number DESC
'''
github.estimate_query_size(q_commits_per_repos_big)
commits_per_pyrepos_big_df=github.query_to_pandas(q_commits_per_repos_big)
commits_per_pyrepos_big_df.head()
commits_per_pyrepos_big_df.shape
commits_per_pyrepos_big_df.describe()
commits_per_pyrepos_big_df_below_100 = commits_per_pyrepos_big_df[(commits_per_pyrepos_big_df.commits_number<100)]
commits_per_pyrepos_big_df_beyond_100 = commits_per_pyrepos_big_df[(commits_per_pyrepos_big_df.commits_number>100)]
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
sns.set()
fig = plt.figure(figsize=(14,6))
# Add subplots
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)

commits_per_pyrepos_big_df_below_100.commits_number.hist(bins=40, linewidth = 0.4, edgecolor='white', ax=ax1);
ax1.set_title('Python commits <100 - distribution per repo ', alpha=0.9, size=14)
# ax1.set_xlim(-10, 10000);
ax1.set_ylabel('Frequency', alpha = 0.8);
ax1.set_xlabel('Commits', alpha = 0.8);

commits_per_pyrepos_big_df_beyond_100.commits_number.hist(bins=200, linewidth = 0.4, edgecolor='white', ax=ax2);
ax2.set_title('Python commits >100 Distribution per repo ', alpha=0.9, size=14)
ax2.set_xlim(-10, 100000);
ax2.set_ylabel('Frequency', alpha = 0.8);
ax2.set_xlabel('Commits', alpha = 0.8);