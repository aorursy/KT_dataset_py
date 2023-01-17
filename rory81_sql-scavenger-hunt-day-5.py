# import package with helper functions 
import bq_helper

# create a helper object for this dataset
github = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                              dataset_name="github_repos")
query = """SELECT l.license,COUNT(sf.path) AS number_of_files
           FROM `bigquery-public-data.github_repos.sample_files` AS sf
           INNER JOIN `bigquery-public-data.github_repos.licenses` AS l
           on sf.repo_name = l.repo_name
           GROUP BY l.license
           ORDER BY number_of_files DESC
        """
file_count_by_license = github.query_to_pandas_safe(query, max_gb_scanned = 6)
print(file_count_by_license)

#How many commits have been made in repos written in Python
commits = """ SELECT sf.repo_name AS repo_name, COUNT(sc.commit) AS num_commits
              FROM `bigquery-public-data.github_repos.sample_commits` AS sc
              INNER JOIN `bigquery-public-data.github_repos.sample_files` AS sf
              ON sc.repo_name = sf.repo_name
              where sf.path LIKE '%.py'
              GROUP BY ROLLUP(repo_name)
              ORDER BY num_commits DESC
          """
commits_per_repo_py = github.query_to_pandas_safe(commits, max_gb_scanned = 6)
print(commits_per_repo_py)