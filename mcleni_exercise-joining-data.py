# import package with helper functions 
import bq_helper

# create a helper object for this dataset
github = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                              dataset_name="github_repos")
# Your code here
github.list_tables()
github.head('sample_commits')
github.head('sample_files')
query = """ SELECT sf.repo_name, COUNT(sc.repo_name) AS n_commits
            FROM (--a subquery to select the python repo names only once from the sample_files table
                  SELECT repo_name
                  FROM `bigquery-public-data.github_repos.sample_files`
                  WHERE path LIKE '%.py'
                  GROUP BY repo_name
                 ) as sf
            INNER JOIN `bigquery-public-data.github_repos.sample_commits` as sc ON sf.repo_name = sc.repo_name
            GROUP BY sf.repo_name
        """
# maybe LEFT JOIN would be a better answer? INNER JOIN excludes those repos with 0 commits
github.estimate_query_size(query)
commits_per_repo = github.query_to_pandas_safe(query, max_gb_scanned = 6)
commits_per_repo
#verifing 
vquery = """ SELECT repo_name, COUNT(repo_name)
            FROM `bigquery-public-data.github_repos.sample_commits` 
            GROUP BY repo_name
            
        """ 
github.estimate_query_size(vquery)
vdf = github.query_to_pandas_safe(vquery, max_gb_scanned = 1)
vdf
vquery2 = """ SELECT repo_name, COUNT(repo_name)
            FROM `bigquery-public-data.github_repos.sample_files` 
            WHERE path LIKE '%.py'
            GROUP BY repo_name
        """ 
github.estimate_query_size(vquery2)
vdf2 = github.query_to_pandas_safe(vquery2, max_gb_scanned = 6)
vdf2[(vdf2.repo_name == 'Microsoft/vscode') | (vdf2.repo_name == 'facebook/react')
    | (vdf2.repo_name == 'apple/swift')| (vdf2.repo_name == 'tensorflow/tensorflow')
    | (vdf2.repo_name == 'torvalds/linux')]
