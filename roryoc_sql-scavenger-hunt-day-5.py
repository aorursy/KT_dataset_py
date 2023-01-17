# import package with helper functions 
import bq_helper

# create a helper object for this dataset
github = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                              dataset_name="github_repos")
# You can use two dashes (--) to add comments in SQL
query = ("""
        -- Select all the columns we want in our joined table
        SELECT L.license, COUNT(sf.path) AS number_of_files
        FROM `bigquery-public-data.github_repos.sample_files` as sf
        -- Table to merge into sample_files
        INNER JOIN `bigquery-public-data.github_repos.licenses` as L 
            ON sf.repo_name = L.repo_name -- what columns should we join on?
        GROUP BY L.license
        ORDER BY number_of_files DESC
        """)

file_count_by_license = github.query_to_pandas_safe(query, max_gb_scanned=6)
# print out all the returned results
print(file_count_by_license)
#github.list_tables()
#['commits',
# 'contents',
# 'files',
# 'languages',
# 'licenses',
# 'sample_commits',
# 'sample_contents',
# 'sample_files',
# 'sample_repos']

#github.table_schema("sample_files")
#[SchemaField('repo_name', 'string', 'NULLABLE', None, ()),
# SchemaField('ref', 'string', 'NULLABLE', None, ()),
# SchemaField('path', 'string', 'NULLABLE', None, ()),
# SchemaField('mode', 'integer', 'NULLABLE', None, ()),
# SchemaField('id', 'string', 'NULLABLE', None, ()),
# SchemaField('symlink_target', 'string', 'NULLABLE', None, ())]

#github.table_schema("files")
#[SchemaField('repo_name', 'string', 'NULLABLE', None, ()),
# SchemaField('ref', 'string', 'NULLABLE', None, ()),
# SchemaField('path', 'string', 'NULLABLE', None, ()),
# SchemaField('mode', 'integer', 'NULLABLE', None, ()),
# SchemaField('id', 'string', 'NULLABLE', None, ()),
# SchemaField('symlink_target', 'string', 'NULLABLE', None, ())]

#github.table_schema("sample_commits")
#[SchemaField('commit', 'string', 'NULLABLE', None, ()),
# SchemaField('tree', 'string', 'NULLABLE', None, ()),
# SchemaField('parent', 'string', 'REPEATED', None, ()),
# SchemaField('author', 'record', 'NULLABLE', None, (SchemaField('name', 'string', 'NULLABLE', None, ()), SchemaField('email', 'string', 'NULLABLE', None, ()), SchemaField('time_sec', 'integer', 'NULLABLE', None, ()), SchemaField('tz_offset', 'integer', 'NULLABLE', None, ()), SchemaField('date', 'timestamp', 'NULLABLE', None, ()))),
# SchemaField('committer', 'record', 'NULLABLE', None, (SchemaField('name', 'string', 'NULLABLE', None, ()), SchemaField('email', 'string', 'NULLABLE', None, ()), SchemaField('time_sec', 'integer', 'NULLABLE', None, ()), SchemaField('tz_offset', 'integer', 'NULLABLE', None, ()), SchemaField('date', 'timestamp', 'NULLABLE', None, ()))),
# SchemaField('subject', 'string', 'NULLABLE', None, ()),
# SchemaField('message', 'string', 'NULLABLE', None, ()),
# SchemaField('trailer', 'record', 'REPEATED', None, (SchemaField('key', 'string', 'NULLABLE', None, ()), SchemaField('value', 'string', 'NULLABLE', None, ()), SchemaField('email', 'string', 'NULLABLE', None, ()))),
# SchemaField('difference', 'record', 'REPEATED', None, (SchemaField('old_mode', 'integer', 'NULLABLE', None, ()), SchemaField('new_mode', 'integer', 'NULLABLE', None, ()), SchemaField('old_path', 'string', 'NULLABLE', None, ()), SchemaField('new_path', 'string', 'NULLABLE', None, ()), SchemaField('old_sha1', 'string', 'NULLABLE', None, ()), SchemaField('new_sha1', 'string', 'NULLABLE', None, ()), SchemaField('old_repo', 'string', 'NULLABLE', None, ()), SchemaField('new_repo', 'string', 'NULLABLE', None, ()))),
# SchemaField('difference_truncated', 'boolean', 'NULLABLE', None, ()),
# SchemaField('repo_name', 'string', 'NULLABLE', None, ()),
# SchemaField('encoding', 'string', 'NULLABLE', None, ())]

#github.table_schema("commits")

github.head("sample_commits")

github.head("sample_contents")
github.head("sample_files")

# Your code goes here :)
# You can use two dashes (--) to add comments in SQL
query = ("""
        -- Select all the columns we want in our joined table
        SELECT f.repo_name, count(1) as commit_count
        FROM 
        (SELECT distinct repo_name
        FROM `bigquery-public-data.github_repos.sample_files` as sf
        WHERE path LIKE '%.py' OR  path LIKE '%.ipynb') f
        inner join `bigquery-public-data.github_repos.sample_commits` as sc on f.repo_name = sc.repo_name
        GROUP BY f.repo_name
        ORDER BY f.repo_name
        """)

commit_count_by_python_repo = github.query_to_pandas_safe(query, max_gb_scanned=6)
commit_count_by_python_repo
query = ("""
        SELECT sf.repo_name, count(1) as python_file_commit_count
        FROM `bigquery-public-data.github_repos.sample_commits` as sf
        WHERE EXISTS (SELECT 1
              FROM UNNEST(difference)
              WHERE new_path LIKE '%.py' OR  new_path LIKE '%.ipynb')
        GROUP BY repo_name
        ORDER BY repo_name
        """)

commit_count_by_python_repo = github.query_to_pandas_safe(query, max_gb_scanned=6)
commit_count_by_python_repo
