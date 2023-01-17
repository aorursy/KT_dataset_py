import bq_helper

git = bq_helper.BigQueryHelper(active_project = "bigquery-public-data",
                              dataset_name   = "github_repos")
query = """WITH f AS
           (SELECT DISTINCT repo_name
           FROM `bigquery-public-data.github_repos.sample_files`
           WHERE path like '%.py'
           )
           
           SELECT f.repo_name, count(c.commit), count(distinct c.commit)
           FROM            `bigquery-public-data.github_repos.sample_commits` as c
                 INNER JOIN f
                 ON c.repo_name = f.repo_name
           GROUP BY f.repo_name
                 
        """

cnt_cmt = git.query_to_pandas_safe(query, max_gb_scanned=30)

cnt_cmt