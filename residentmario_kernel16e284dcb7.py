from google.cloud import bigquery

client = bigquery.Client()



QUERY = (

"""SELECT REGEXP_EXTRACT(content, r'(random_state=\d*|seed=\d*|random_seed=\d*|random_number=\d*)') FROM `bigquery-public-data.github_repos.sample_contents`"""

)



query_job = client.query(QUERY)



iterator = query_job.result(timeout=30)

rows = list(iterator)
len(rows)
import pandas as pd

seeds = pd.Series(rows)
import numpy as np

seeds = (

    seeds.map(lambda s: s[0])

     .map(lambda s: np.nan if pd.isnull(s) else s.split("=")[-1].strip())

     .map(lambda s: np.nan if (pd.isnull(s) or s == "") else float(s))

     .value_counts(dropna=False)

)
import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')



(seeds

     # Reagg by percentage usage, excluding the empty (NaN) seed.

     .pipe(lambda srs: srs / srs.iloc[1:].sum())

     .sort_values(ascending=False)

     # Exclude the empty (NaN) seed.

     .head(11)

     .iloc[1:]

     .pipe(lambda srs: srs.reindex(srs.index.astype(int)))

     .plot.bar(

         figsize=(12, 6), title='Most Common Integer Random Seeds'

     )

)

ax = plt.gca()

ax.set_ylabel('Percent Usage')

ax.set_xlabel('Seed (Integer)')

pass