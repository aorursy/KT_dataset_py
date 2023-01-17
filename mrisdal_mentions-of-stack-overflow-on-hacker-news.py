from google.cloud import bigquery

import pandas as pd



client = bigquery.Client()



# Using WHERE reduces the amount of data scanned / quota used

query = """

SELECT title, time_ts

FROM `bigquery-public-data.hacker_news.stories`

WHERE REGEXP_CONTAINS(title, r"Stack Overflow")

ORDER BY time

"""



query_job = client.query(query)



iterator = query_job.result(timeout=30)

rows = list(iterator)



# Transform the rows into a nice pandas dataframe

headlines = pd.DataFrame(data=[list(x.values()) for x in rows], columns=list(rows[0].keys()))



# Look at the first 10 headlines

headlines.head(10)
import wordcloud

import matplotlib.pyplot as plt



words = ' '.join(headlines.title).lower()

cloud = wordcloud.WordCloud(background_color='black',

                            max_font_size=200,

                            width=1600,

                            height=800,

                            max_words=300,

                            relative_scaling=.5).generate(words)

plt.figure(figsize=(20,10))

plt.axis('off')

plt.savefig('kaggle-hackernews.png')

plt.imshow(cloud);