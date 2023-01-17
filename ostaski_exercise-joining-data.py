# import package with helper functions 
import bq_helper

# create a helper object for this dataset
github = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                              dataset_name="github_repos")
query = ("""
        -- Select the count of files in sample_files that are written in Python
        SELECT COUNT(sc.commit) AS number_of_commits, sf.repo_name
        FROM `bigquery-public-data.github_repos.sample_files` AS sf
        -- merge sample_commits into sample_files
        INNER JOIN `bigquery-public-data.github_repos.sample_commits` AS sc 
            ON sf.repo_name = sc.repo_name
        WHERE sf.path LIKE '%.py'
        GROUP BY sf.repo_name
        ORDER BY number_of_commits DESC
        """)

# estimate query size
github.estimate_query_size(query)
# 5.271098022349179,  so bump the max_gb_scanned in query up to 6 gb

# run a "safe" query and store the resultset into a dataframe
python_commits = github.query_to_pandas_safe(query, max_gb_scanned=6)

# taking a look
print(python_commits)

# saving this in case we need it later
python_commits.to_csv("python_commits.csv")
