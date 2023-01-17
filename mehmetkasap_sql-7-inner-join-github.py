# import package with helper functions 
import bq_helper

# create a helper object for this dataset
github = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                              dataset_name="github_repos")
github.head('sample_commits',1)
github.head('sample_files',1)
query = ''' SELECT  sf.id AS ID,
                    COUNT(sc.commit) AS commit                   
            FROM `bigquery-public-data.github_repos.sample_files` as sf
            INNER JOIN `bigquery-public-data.github_repos.sample_commits` as sc
                        ON sf.repo_name = sc.repo_name
            GROUP BY ID
        '''
commit_repos = github.query_to_pandas_safe(query, max_gb_scanned=20)
commit_repos.head(10)
github.head('sample_files',1)
queryy = ''' SELECT repo_name, id
             FROM `bigquery-public-data.github_repos.sample_files`
             WHERE path LIKE '%.py'
         '''
id_py = github.query_to_pandas(queryy)
id_py.head()
query2 = ''' WITH path_py AS
             (
                 SELECT repo_name, id
                 FROM `bigquery-public-data.github_repos.sample_files`
                 WHERE path LIKE '%.py'
             )
             
             SELECT 
                     py.id as ID, 
                     COUNT(sc.commit) as COMMIT 
             FROM path_py as py
             INNER JOIN `bigquery-public-data.github_repos.sample_commits` as sc
                         ON py.repo_name = sc.repo_name
             GROUP BY ID
             ORDER BY COMMIT
         '''
path_py_commit_id = github.query_to_pandas(query2)
path_py_commit_id