import pandas as pd

# https://github.com/SohierDane/BigQuery_Helper

from bq_helper import BigQueryHelper
bq_assistant = BigQueryHelper("bigquery-public-data", "github_repos")
%%time

bq_assistant.list_tables()
%%time

bq_assistant.table_schema("licenses")
%%time

bq_assistant.head("licenses", num_rows=10)
QUERY = """

        SELECT message

        FROM `bigquery-public-data.github_repos.commits`

        WHERE LENGTH(message) > 6 AND LENGTH(message) <= 20

        LIMIT 2000

        """
%%time

bq_assistant.estimate_query_size(QUERY)
QUERY = """

        SELECT message

        FROM `bigquery-public-data.github_repos.commits`

        WHERE LENGTH(message) > 6 AND LENGTH(message) <= 20

        LIMIT 4000 -- twice as many commit messages

        """
%%time

bq_assistant.estimate_query_size(QUERY)
%%time

QUERY = """

        SELECT message

        FROM `bigquery-public-data.github_repos.commits`

        """
%%time

bq_assistant.estimate_query_size(QUERY)
QUERY = """

        SELECT message

        FROM `bigquery-public-data.github_repos.commits`

        WHERE LENGTH(message) > 6 AND LENGTH(message) <= 20

        LIMIT 2000

        """
%%time

df = bq_assistant.query_to_pandas_safe(QUERY)
QUERY = """

        SELECT license, COUNT(*) AS count

        FROM `bigquery-public-data.github_repos.licenses`

        GROUP BY license

        ORDER BY COUNT(*) DESC

        """
%%time

bq_assistant.estimate_query_size(QUERY)
%%time

df = bq_assistant.query_to_pandas_safe(QUERY)
print('Size of dataframe: {} Bytes'.format(int(df.memory_usage(index=True, deep=True).sum())))
#df.head()

df
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