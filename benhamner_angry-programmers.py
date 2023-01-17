from google.cloud import bigquery

import pandas as pd



client = bigquery.Client()



query = """

SELECT message

FROM `bigquery-public-data.github_repos.commits`

WHERE LENGTH(message) > 10 AND LENGTH(message) <= 100

  AND message LIKE 'fuck%'

LIMIT 50

"""



query_job = client.query(query)



iterator = query_job.result(timeout=30)

rows = list(iterator)



pd.DataFrame({"message": [row.message.strip() for row in rows]})