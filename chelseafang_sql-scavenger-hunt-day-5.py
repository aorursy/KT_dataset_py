import bq_helper as bq

github=bq.BigQueryHelper(active_project="bigquery-public-data",
                         dataset_name='github_repos'
                        )

query1="""
        SELECT l.license, COUNT(sf.path) AS number_of_files
        FROM `bigquery-public-data.github_repos.sample_files` AS sf
        
        INNER JOIN `bigquery-public-data.github_repos.licenses` AS l
        ON sf.repo_name=l.repo_name
        
        GROUP BY l.license
        ORDER BY number_of_files DESC
"""
result1=github.query_to_pandas_safe(query1,max_gb_scanned=21)
print(result1)
query2="""
        WITH temp AS
        (
            SELECT DISTINCT repo_name
            FROM `bigquery-public-data.github_repos.sample_files`
            WHERE path LIKE '%.py'
        )
        
        SELECT sc.repo_name, COUNT(sc.commit) AS commit_number
        FROM `bigquery-public-data.github_repos.sample_commits` AS sc
        
        INNER JOIN temp
        ON temp.repo_name=sc.repo_name
        
        GROUP BY sc.repo_name
        ORDER BY commit_number
"""


result2=github.query_to_pandas_safe(query2,max_gb_scanned=21)
print(result2)
import matplotlib.pyplot as plt
plt.barh(result2.repo_name,result2.commit_number,log=True)

