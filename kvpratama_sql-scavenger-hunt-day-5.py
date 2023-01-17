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
python_count_query = ("""
        SELECT sf.repo_name, COUNT(sc.commit) AS number_of_commit
        FROM `bigquery-public-data.github_repos.sample_commits` as sc
        INNER JOIN `bigquery-public-data.github_repos.sample_files` as sf 
            ON sc.repo_name = sf.repo_name
        WHERE sf.path LIKE '%.py'
        GROUP BY sf.repo_name
        ORDER BY number_of_commit DESC
        """)

python_count = github.query_to_pandas_safe(python_count_query, max_gb_scanned=6)
print(python_count)
check_commit_query = ("""
        SELECT repo_name, COUNT(commit) AS no_of_commit
        FROM `bigquery-public-data.github_repos.sample_commits`
        WHERE repo_name = 'torvalds/linux' OR repo_name = 'tensorflow/tensorflow' OR 
            repo_name = 'apple/swift' OR repo_name = 'facebook/react' OR 
            repo_name = 'Microsoft/vscode'
        GROUP BY repo_name
        ORDER BY repo_name
        """)
check_commit_count = github.query_to_pandas_safe(check_commit_query)
print(check_commit_count)
check_py_file_query = ("""
        SELECT repo_name, COUNT(path) AS no_of_py_file
        FROM `bigquery-public-data.github_repos.sample_files`
        WHERE (repo_name = 'torvalds/linux' OR repo_name = 'tensorflow/tensorflow' OR 
            repo_name = 'apple/swift' OR repo_name = 'facebook/react' OR 
            repo_name = 'Microsoft/vscode') AND path LIKE '%.py'
        GROUP BY repo_name
        ORDER BY repo_name
        """)
check_py_file = github.query_to_pandas_safe(check_py_file_query, max_gb_scanned=6)
print(check_py_file)
check_commit_count = check_commit_count.assign(number_of_py_file=check_py_file['no_of_py_file'])
check_commit_count = check_commit_count.assign(commit_x_file=check_commit_count['no_of_commit'].multiply(check_py_file['no_of_py_file']))
print(check_commit_count.sort_values(by='commit_x_file', ascending=False))
# Repository that has python file
python_repo_query = ("""
        SELECT DISTINCT repo_name
        FROM `bigquery-public-data.github_repos.sample_files`
        WHERE path LIKE '%.py'
        ORDER BY repo_name
        """)
python_repo = github.query_to_pandas_safe(python_repo_query, max_gb_scanned=6)
python_repo
# No. of Commit of Repository with python file
python_repo_commit_query = ("""
        WITH python_repo AS (
            SELECT DISTINCT repo_name
                FROM `bigquery-public-data.github_repos.sample_files`
                WHERE path LIKE '%.py'
        )
        SELECT pr.repo_name, COUNT(sc.commit) AS number_of_commit
        FROM  python_repo as pr
        INNER JOIN `bigquery-public-data.github_repos.sample_commits` as sc
            ON pr.repo_name = sc.repo_name
        GROUP BY pr.repo_name
        ORDER BY number_of_commit DESC
        """)
python_commit = github.query_to_pandas_safe(python_repo_commit_query, max_gb_scanned=6)
python_commit