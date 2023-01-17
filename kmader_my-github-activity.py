import pandas as pd
import bq_helper
github_repos = bq_helper.BigQueryHelper(active_project= "bigquery-public-data", 
                                       dataset_name = "github_repos")
github_repos.list_tables()
# print information on all the columns in the "commits" table
# in the github_repos dataset
github_repos.table_schema("commits")
commit_head = github_repos.head("commits", start_index = 9999) # starting at the beginning is boring
commit_head
commit_head['committer'].values[0]
my_repos_query = """SELECT repository, commit.author.name, COUNT(*) as count
            FROM `bigquery-public-data.github_repos.commits` commit
            CROSS JOIN UNNEST(commit.repo_name) as repository
            WHERE STARTS_WITH(upper(repository), upper("kmader"))
            GROUP BY commit.author.name, repository
        """
print('%2.2f GB, query' % github_repos.estimate_query_size(my_repos_query))
my_repo_df = github_repos.query_to_pandas_safe(my_repos_query, max_gb_scanned=110)
print(my_repo_df.shape[0], 'total entries')
my_repo_df.sort_values('count', ascending = False).head(10)
query_simple= """SELECT repository, commit.author.email, COUNT(*) as count
            FROM `bigquery-public-data.github_repos.commits` commit
            CROSS JOIN UNNEST(commit.repo_name) as repository
            WHERE repository = "kmader/Quantitative-Big-Imaging-2018"
            GROUP BY commit.author.email, repository
        """
github_repos.estimate_query_size(query_simple)
commit_summary = github_repos.query_to_pandas_safe(query_simple, max_gb_scanned=110)
commit_summary.head()
query_me = """SELECT commit.author.name, commit.author.date, commit.message, repository
            FROM `bigquery-public-data.github_repos.commits` commit
            CROSS JOIN UNNEST(commit.repo_name) as repository
            WHERE author.email = 'kevinmader@gmail.com' OR 
                author.name = 'Kevin Mader' OR 
                author.email = 'kevin.mader@gmail.com' OR
                author.name = 'kmader' OR
                author.email = 'kmader@users.noreply.github.com' OR
                committer.email = 'kmader@users.noreply.github.com'
        """
github_repos.estimate_query_size(query_me)
my_commits = github_repos.query_to_pandas_safe(query_me, max_gb_scanned=1000)
print(my_commits.shape[0], 'rows found')
my_commits.head()
my_commits['repo_name'] = my_commits['repository'].map(lambda x: x.split('/')[-1])
my_commits[['repository', 'date', 'message']].sample(10)
my_commits.groupby('repo_name').apply(lambda x: pd.Series({'count': x.shape[0]})).reset_index().sort_values('count', ascending = False)
f = my_commits.groupby('date').agg({'name': 'count'}).reset_index()
f.plot('date', 'name')
my_commits.sort_values('date', ascending = False).head(50)
my_commits.sort_values('date', ascending = False).sample(10)
