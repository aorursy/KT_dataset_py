import bq_helper
github = bq_helper.BigQueryHelper(active_project = "bigquery-public-data",dataset_name="github_repos")

query = """ select L.license,count(o.path) as number_of_files from 
            `bigquery-public-data.github_repos.sample_files` as o
            inner join `bigquery-public-data.github_repos.licenses` as L 
            on o.repo_name = L.repo_name
            group by license
            order by number_of_files DESC
            """
github.estimate_query_size(query)
ans = github.query_to_pandas_safe(query,max_gb_scanned=6)
ans.head()
query2 = """ select count(path) from `bigquery-public-data.github_repos.sample_files`
            where path LIKE '%.py'
            """
github.estimate_query_size(query2)
github.query_to_pandas_safe(query2,max_gb_scanned = 4)
LastQuery = """ WITH gits_py as
                (
                    select distinct repo_name from `bigquery-public-data.github_repos.sample_files` as p
                    where p.path LIKE '%.py'
                )
                select o.repo_name , count(o.commit) as CountCom 
                from `bigquery-public-data.github_repos.sample_commits` as o
                join gits_py on o.repo_name = gits_py.repo_name
                group by o.repo_name
                order by CountCom DESC
             """
github.estimate_query_size(LastQuery)
res = github.query_to_pandas_safe(LastQuery,max_gb_scanned = 6)
print(res)
