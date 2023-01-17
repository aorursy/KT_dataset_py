import bq_helper

import pandas as pd



BQ_TABLES = "bigquery-public-data"

DATASET_NAME="github_repos"

GITHUB_REPOS = bq_helper.BigQueryHelper(active_project=BQ_TABLES, dataset_name=DATASET_NAME)
QUERY_LICENSES = ("SELECT * "

        "FROM `bigquery-public-data.github_repos.licenses` ")



GITHUB_REPOS.estimate_query_size(QUERY_LICENSES) # in GB
licenses = GITHUB_REPOS.query_to_pandas_safe(QUERY_LICENSES, max_gb_scanned=15)

licenses.to_csv("licenses.csv")
QUERY_LANGUAGES = ("SELECT * "

        "FROM `bigquery-public-data.github_repos.languages` ")



GITHUB_REPOS.estimate_query_size(QUERY_LANGUAGES) # in GB
languages = GITHUB_REPOS.query_to_pandas_safe(QUERY_LANGUAGES, max_gb_scanned=15)

languages.to_csv("languages.csv")
QUERY_SAMPLE_COMMITS = ("SELECT * "

        "FROM `bigquery-public-data.github_repos.sample_commits` "

        "LIMIT 10000")



GITHUB_REPOS.estimate_query_size(QUERY_SAMPLE_COMMITS) # in GB
commits = GITHUB_REPOS.query_to_pandas_safe(QUERY_SAMPLE_COMMITS, max_gb_scanned=15)

commits.to_csv("sample_commits_10000.csv")
QUERY_SAMPLE_CONTENTS = ("SELECT * "

        "FROM `bigquery-public-data.github_repos.sample_contents` "

        "LIMIT 10000")



GITHUB_REPOS.estimate_query_size(QUERY_SAMPLE_CONTENTS) # in GB
contents = GITHUB_REPOS.query_to_pandas_safe(QUERY_SAMPLE_CONTENTS, max_gb_scanned=30)

contents.to_csv("sample_contents_10000.csv")
QUERY_SAMPLE_FILES = ("SELECT * "

        "FROM `bigquery-public-data.github_repos.sample_files` "

        "LIMIT 10000")



GITHUB_REPOS.estimate_query_size(QUERY_SAMPLE_FILES) # in GB
files = GITHUB_REPOS.query_to_pandas_safe(QUERY_SAMPLE_FILES, max_gb_scanned=15)

files.to_csv("sample_files_10000.csv")
QUERY_SAMPLE_REPOS = ("SELECT * "

        "FROM `bigquery-public-data.github_repos.sample_repos` ")



GITHUB_REPOS.estimate_query_size(QUERY_SAMPLE_REPOS) # in GB
repos = GITHUB_REPOS.query_to_pandas_safe(QUERY_SAMPLE_REPOS, max_gb_scanned=15)

repos.to_csv("sample_repos.csv")