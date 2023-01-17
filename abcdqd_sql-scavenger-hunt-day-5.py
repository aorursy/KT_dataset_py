import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import bq_helper as bq

gh = bq.BigQueryHelper(active_project = 'bigquery-public-data',
                       dataset_name = 'github_repos')
gh.head('sample_commits')
gh.head('sample_files')
# use sample commits and files for this query
query = """
        with commit_table as
        (select
            repo_name,
            count(commit) as num_commits
        from `bigquery-public-data.github_repos.sample_commits`
        group by repo_name),
        python_repo_table as
        (select
            repo_name,
            count(path) as num_files
        from `bigquery-public-data.github_repos.sample_files`
        where path like '%.py'
        group by repo_name)
        select
            a.repo_name,
            a.num_commits,
            b.num_files
        from commit_table a
        inner join python_repo_table b
        on a.repo_name = b.repo_name
        order by num_commits desc
        """

gh.query_to_pandas(query)
# explore full tables
query = """
        select
            repo_name,
            count(path) as num_files
        from `bigquery-public-data.github_repos.files`
        where path like '%.py'
        group by repo_name
        """
test_query_0 = gh.query_to_pandas(query)
test_query_0.head()
# query = """
#        select
#            repo_name,
#            count(commit) as num_commits
#        from `bigquery-public-data.github_repos.commits`
#        group by repo_name
#        """
# gh.query_to_pandas(query) # grouping of expressions of type array is not allowed

gh.head('commits') # repo_name is an array type - commits to multiple repos?
query = """
        select 
            flattened_repo,
            count(flattened_repo) as num_commits
        from `bigquery-public-data.github_repos.commits` commit_table
        cross join unnest(commit_table.repo_name) as flattened_repo
        group by flattened_repo
        order by num_commits desc
        """
test_query = gh.query_to_pandas(query)
test_query.head()
query = """
        with commit_table as
        (select 
            flattened_repo,
            count(flattened_repo) as num_commits
        from `bigquery-public-data.github_repos.commits` commit_table
        cross join unnest(commit_table.repo_name) as flattened_repo
        group by flattened_repo),
        python_table as
        (select
            repo_name,
            count(path) as num_files
        from `bigquery-public-data.github_repos.files`
        where path like '%.py'
        group by repo_name)
        select 
            a.flattened_repo as repo_name,
            a.num_commits,
            b.num_files
        from commit_table a
        inner join python_table b
        on a.flattened_repo = b.repo_name
        order by num_commits desc
        """

full_table = gh.query_to_pandas(query)
full_table.head(n = 10)