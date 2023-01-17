from google.cloud import bigquery

client = bigquery.Client()



def count_file_pattern(pattern: str):

    query = ("""

        SELECT COUNT(path) AS total_files

        FROM `bigquery-public-data.github_repos.files`

        WHERE REGEXP_CONTAINS(path, r'(^|/)%s$')

        """ % pattern)

    query_job = client.query(query)

    iterator = query_job.result(timeout=30)

    rows = list(iterator)

    return rows[0].total_files             
# "test_file.py" style

count_file_pattern(r'test_[^\./]+.py')
# "file_test.py" style

count_file_pattern(r'[^/]+_test.py')
# "testfile.py" style

count_file_pattern(r'test[a-zA-Z0-9][^_/]*.py')
# "filetest.py" style

count_file_pattern(r'[^_/]*[a-zA-Z0-9]+test.py')