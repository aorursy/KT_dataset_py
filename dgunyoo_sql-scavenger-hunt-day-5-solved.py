import bq_helper
github = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                              dataset_name="github_repos")
query = """
        WITH repos AS(
            SELECT distinct repo_name
            FROM `bigquery-public-data.github_repos.sample_files`
            WHERE path LIKE '%.py')
        SELECT sc.repo_name, count(sc.commit) AS number_of_commits
        FROM `bigquery-public-data.github_repos.sample_commits` as sc
        INNER JOIN repos
            ON sc.repo_name = repos.repo_name
        
        GROUP BY sc.repo_name
        ORDER BY number_of_commits DESC
        """
print(github.query_to_pandas(query))