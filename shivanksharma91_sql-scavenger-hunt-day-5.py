#importing helper package
import bq_helper

#setting a helper object
github=bq_helper.BigQueryHelper(active_project='bigquery-public-data',
                                dataset_name='github_repos')

query1='''SELECT COUNT(scommit.commit) as Nbr_Of_Commits ,sfiles.repo_name as Repo_Name
          FROM `bigquery-public-data.github_repos.sample_commits` AS scommit
          INNER JOIN `bigquery-public-data.github_repos.sample_files`AS sfiles
          ON scommit.repo_name=sfiles.repo_name
          WHERE sfiles.path LIKE '%.py'
          GROUP BY 2
          ORDER BY 1 DESC
       '''

commits_per_repo=github.query_to_pandas_safe(query1,max_gb_scanned=6)

# print out all the returned results
print(commits_per_repo)

