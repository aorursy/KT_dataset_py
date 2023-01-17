import bq_helper

github = bq_helper.BigQueryHelper(active_project='bigquery-public-data',
                                 dataset_name='github_repos')
github.head('sample_files')
github.head('licenses', num_rows = 10)
query = ("""
        -- Select all the columns we want in our joined table
        SELECT
            L.license, 
            COUNT(sf.path) AS number_of_files
        FROM `bigquery-public-data.github_repos.sample_files` as sf
        
        -- Table to merge into sample_files
        INNER JOIN `bigquery-public-data.github_repos.licenses` as L 
            
            -- what columns should we join on?
            ON sf.repo_name = L.repo_name 
        
        GROUP BY L.license
        ORDER BY number_of_files DESC
        """)

file_count_by_license = github.query_to_pandas_safe(query, max_gb_scanned=6)
file_count_by_license
github.head('sample_commits')
query1 = """
            WITH python_repos AS (
                SELECT DISTINCT repo_name -- Notice DISTINCT
                FROM `bigquery-public-data.github_repos.sample_files`
                WHERE path LIKE '%.py')
            SELECT commits.repo_name, COUNT(commit) AS num_commits
            FROM `bigquery-public-data.github_repos.sample_commits` AS commits
            JOIN python_repos
                ON  python_repos.repo_name = commits.repo_name
            GROUP BY commits.repo_name
            ORDER BY num_commits DESC
        """

python_number = github.query_to_pandas_safe(query1, max_gb_scanned=10)
python_number = github.query_to_pandas(query1)
print(python_number)
