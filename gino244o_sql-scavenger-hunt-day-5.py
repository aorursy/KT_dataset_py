import bq_helper
# bigquery-public-data.github_repos
github = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                              dataset_name="github_repos")
github.list_tables()
github.table_schema('sample_commits')
github.head('sample_commits')
github.table_schema('sample_files')
github.head('sample_files')
github.head('sample_repos')
querypy = """
select
    count(*)
from
    `bigquery-public-data.github_repos.sample_files`
where
    path like '%.py'
"""
github.estimate_query_size(querypy)
github.query_to_pandas(querypy)
queryquestion = """
select
    count(a.repo_name)
from (
        select
            repo_name
        from
            `bigquery-public-data.github_repos.sample_commits`
    ) a
    inner join (
        select
            repo_name
        from
            `bigquery-public-data.github_repos.sample_files`
        where
            path like '%.py'
    ) b
      on a.repo_name = b.repo_name
"""
github.estimate_query_size(queryquestion)
github.query_to_pandas(queryquestion)