import collections
import bq_helper
fields = ['commits',
          'contents',
          'files',
          'languages',
          'sample_commits',
          'sample_contents',
          'sample_files',
          'sample_repos']
mapping = {s: '`bigquery-public-data.github_repos.{}`'.format(s) for s in fields}
db = collections.namedtuple('db', fields)(**mapping)
# Query example
query = f"SELECT * FROM {db.sample_repos}"
gh = bq_helper.BigQueryHelper(active_project='bigquery-public-data',
                              dataset_name='github_repos')
gh.estimate_query_size(query)