%%time
usernames = "asdasd|asdasd|Asdasd|asdasd"
QUERY = r"""
SELECT 
c.id,
f.repo_name AS repo,
f.path AS path,
match
FROM `bigquery-public-data.github_repos.contents` c
CROSS JOIN UNNEST(REGEXP_EXTRACT_ALL(CONCAT('\n', c.content,'\n'), r'^.*:[\s]*"[-\w.]+/.+".*$')) AS match
INNER JOIN `bigquery-public-data.github_repos.files` f ON c.id=f.id
WHERE c.binary=False AND (ENDS_WITH(f.path, 'package.json') OR ENDS_WITH(f.path, 'bower.json')) AND NOT ENDS_WITH(f.path, '.md') AND NOT ENDS_WITH(f.path, '.html') AND match IS NOT NULL
"""
print(QUERY)

with open("/src/bq-helper/bq_helper.py", "r") as f:
    lines = f.readlines()
with open("/src/bq-helper/bq_helper3.py", "w") as f:
    for line in lines:
        if line.strip() != "rows = list(query_job.result(timeout=30))":
            f.write(line)
        else:
            f.write("         rows = list(query_job.result())\n")
from bq_helper3 import BigQueryHelper
bq_assistant = BigQueryHelper("bigquery-public-data", "github_repos")
df = bq_assistant.query_to_pandas(QUERY)

from IPython.display import HTML
import pandas as pd
import numpy as np

df.to_csv('resultsFinalMatch.csv')

df.head(10)
from IPython.display import HTML
import pandas as pd
import numpy as np

df.to_csv('resultsFinalMatch.csv')

def create_download_link(title = "Download CSV file", filename = "data.csv"):  
    html = '<a href={filename}>{title}</a>'
    html = html.format(title=title,filename=filename)
    return HTML(html)

# create a link to download the dataframe which was saved with .to_csv method
create_download_link(filename='resultsFinalMatch.csv')

print(len(df))
%%time
bq_assistant.table_schema("files")
%%time
bq_assistant.head("files", num_rows=10)