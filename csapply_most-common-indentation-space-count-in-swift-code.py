import pandas as pd
from google.cloud import bigquery
client = bigquery.Client()
QUERY = ('''
#standardSQL
WITH
  lines AS (
  SELECT
    SPLIT(content, '\\n') AS line,
    id
  FROM
    `bigquery-public-data.github_repos.sample_contents`
  WHERE
    sample_path LIKE "%.swift" )
SELECT
  space_count,
  COUNT(space_count) AS number_of_occurence
FROM (
  SELECT
    id,
    MIN(CHAR_LENGTH(REGEXP_EXTRACT(flatten_line, r"^ +"))) AS space_count
  FROM
    lines
  CROSS JOIN
    UNNEST(lines.line) AS flatten_line
  WHERE
    REGEXP_CONTAINS(flatten_line, r"^ +")
  GROUP BY
    id )
GROUP BY
  space_count
ORDER BY
  number_of_occurence DESC
''')

query_job = client.query(QUERY)

iterator = query_job.result(timeout=30)
rows = list(iterator)
rows = [dict(row) for row in rows]
df = pd.DataFrame(rows)
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
df[:6].plot(kind='bar', x='space_count', y='number_of_occurence')

ax = plt.gca()
ax.set_ylabel('Number of Occurence')
ax.set_xlabel('Indentation Space Count')
pass