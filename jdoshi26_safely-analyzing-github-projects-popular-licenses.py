import pandas as pd

# https://github.com/SohierDane/BigQuery_Helper

from bq_helper import BigQueryHelper
bq_assistant = BigQueryHelper("bigquery-public-data", "github_repos")
bq_assistant.list_tables()
bq_assistant.table_schema("licenses")
bq_assistant.head("licenses", num_rows=10)
QUERY = """

        SELECT message

        FROM `bigquery-public-data.github_repos.commits`

        WHERE LENGTH(message) > 6 AND LENGTH(message) <= 20

        LIMIT 2000

        """
bq_assistant.estimate_query_size(QUERY)
QUERY = """

        SELECT message

        FROM `bigquery-public-data.github_repos.commits`

        WHERE LENGTH(message) > 6 AND LENGTH(message) <= 20

        LIMIT 4000 -- twice as many commit messages

        """
bq_assistant.estimate_query_size(QUERY)
QUERY = """

        SELECT message

        FROM `bigquery-public-data.github_repos.commits`

        """
bq_assistant.estimate_query_size(QUERY)
QUERY = """

        SELECT message

        FROM `bigquery-public-data.github_repos.commits`

        WHERE LENGTH(message) > 6 AND LENGTH(message) <= 20

        LIMIT 2000

        """
df = bq_assistant.query_to_pandas_safe(QUERY)
QUERY = """

        SELECT license, COUNT(*) AS count

        FROM `bigquery-public-data.github_repos.licenses`

        GROUP BY license

        ORDER BY COUNT(*) DESC

        """
bq_assistant.estimate_query_size(QUERY)
df = bq_assistant.query_to_pandas_safe(QUERY)
print('Size of dataframe: {} Bytes'.format(int(df.memory_usage(index=True, deep=True).sum())))
df.head()
df.shape
import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

plt.style.use('ggplot')
sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5})



f, g = plt.subplots(figsize=(12, 9))

g = sns.barplot(x="license", y="count", data=df, palette="Blues_d")

g.set_xticklabels(g.get_xticklabels(), rotation=30)

plt.title("Popularity of Licenses Used by Open Source Projects on GitHub")

plt.show(g)