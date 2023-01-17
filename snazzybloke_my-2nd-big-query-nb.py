# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
from bq_helper import BigQueryHelper
bq_assist = BigQueryHelper('bigquery-public-data', 'github-repos')
bq_assist.list_tables()
bq_assist.table_schema('licences')
bq_assist.head('licenses', num_rows=10)
query2= """
        SELECT message
        FROM `bigquery-public-data.github_repos.commits`
        WHERE LENGTH(message) > 6 AND LENGTH(message) <= 20
        LIMIT 2000
        """
bq_assist.estimate_query_size(query2)
query3 = """
         SELECT message
         FROM `bigquery-public-data.github_repos.commits`
         """
bq_assist.estimate_query_size(query3)
df = bq_assist.query_to_pandas_safe(query2)
#What are the most pupular licenses for open source projects shared on github?
QUERY = """
        SELECT license, COUNT(*) AS count
        FROM `bigquery-public-data.github_repos.licenses`
        GROUP BY license
        ORDER BY COUNT(*) DESC
        """
bq_assist.estimate_query_size(QUERY)
df = bq_assist.query_to_pandas_safe(QUERY)
df.memory_usage(index=True, deep=True).sum()
%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('ggplot')
sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5})
f, g = plt.subplots(figsize=(12, 9))
g = sns.barplot(x='license', y='count', data=df, palette='Blues_d')
g.set_xticklabels(g.get_xticklabels(), rotation=30)
plt.title('Popularity of open source project licenses on github')
plt.show(g)
