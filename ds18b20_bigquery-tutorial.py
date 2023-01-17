# Reference: https://www.kaggle.com/mrisdal/mentions-of-kaggle-on-hacker-news
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from google.cloud import bigquery

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
client = bigquery.Client()
# Using WHERE reduces the amount of data scanned / quota used
query = """
SELECT title, time_ts
FROM `bigquery-public-data.hacker_news.stories`
WHERE REGEXP_CONTAINS(title, r"(k|K)aggle")
ORDER BY time
"""
query_job = client.query(query)
iterator = query_job.result(timeout=30)
rows = list(iterator)
# Transform the rows into a nice pandas dataframe
# [k for k in rows[0].keys()]
headlines = pd.DataFrame(data=[list(r.values()) for r in rows], columns=[key for key in rows[0].keys()])
headlines.shape
# check the first 10 lines
headlines.head(10)
import wordcloud
import matplotlib.pyplot as plt

words = ' '.join(headlines.title).lower()
cloud = wordcloud.WordCloud(background_color='white',
                            max_font_size=200,
                            width=1600,
                            height=800,
                            max_words=300,
                            relative_scaling=.5).generate(words)
plt.figure(figsize=(20,10))
plt.axis('off')
plt.savefig('kaggle-hackernews.png')
plt.imshow(cloud)