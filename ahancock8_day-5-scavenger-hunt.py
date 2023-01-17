import bq_helper
github = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                              dataset_name="github_repos")
github.list_tables()
github.head("sample_commits")
github.head("sample_files")

snake= """select count(c.commit) as commits
        FROM `bigquery-public-data.github_repos.sample_files` as sf
        inner join `bigquery-public-data.github_repos.sample_commits`as c
        on sf.repo_name = c.repo_name
        where sf.path like '%.py'
     
        """
github.estimate_query_size(snake)

Python_Commits= github.query_to_pandas_safe(snake, max_gb_scanned=7)

Python_Commits