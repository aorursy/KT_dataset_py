from google.cloud import bigquery

import pandas as pd



client = bigquery.Client()



# Using WHERE reduces the amount of data scanned / quota used

query = """

SELECT 

    EXTRACT(YEAR FROM time_ts) AS year,

    COUNT(*) count

FROM `bigquery-public-data.hacker_news.comments`

WHERE REGEXP_CONTAINS(LOWER(text), r"well(, | )actually")

GROUP BY year

ORDER BY year

"""



query_job = client.query(query)



iterator = query_job.result(timeout=30)

rows = list(iterator)



# Transform the rows into a nice pandas dataframe

actually = pd.DataFrame(data=[list(x.values()) for x in rows], columns=list(rows[0].keys()))



# Look at the first 10 headlines

actually.head(10)
# Plot the well actuallies ...

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

plt.style.use('ggplot')

sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5})



f, g = plt.subplots(figsize=(15, 9))

g = sns.lineplot(x="year", y="count", data=actually, palette="Blues_d")

plt.title("Well actualies on Hacker News")

plt.show(g);