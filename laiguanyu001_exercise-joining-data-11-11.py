# import package with helper functions 
import bq_helper

# create a helper object for this dataset
github = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                              dataset_name="github_repos")
# Your code here
github.head("sample_files")

github.head("sample_commits")
#How many commits (recorded in the "sample_commits" table)
#have been made in repos written in the Python programming language? 
#(I'm looking for the number of commits per repo
#for all the repos written in Python.)
query = """select sf.repo_name, count(commit) as number_of_commit
        from `bigquery-public-data.github_repos.sample_commits` as sc
        inner join `bigquery-public-data.github_repos.sample_files` as sf
        on sf.repo_name = sc.repo_name
        where sf.path like '%.py'
        group by repo_name
        order by number_of_commit desc
      

"""
output = github.query_to_pandas_safe(query, max_gb_scanned=6)
print(output)