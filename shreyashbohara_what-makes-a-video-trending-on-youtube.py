import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import scipy.stats as stats



videos_usa = pd.read_csv("../input/USvideos.csv")

videos_gb = pd.read_csv("../input/GBvideos.csv")
videos_usa.head()
videos_usa = videos_usa.drop_duplicates(subset='title',keep='first')
videos_usa.sort_values(by='views',ascending=False)
videos_usa_top10 = videos_usa.nlargest(10, 'views')
videos_usa_top10.head()
plt.figure(figsize=(13, 7), dpi=200)

plt.bar(videos_usa_top10['title'],videos_usa_top10['views'])

plt.xticks(rotation=90)

plt.ylabel('views')

plt.title('Number of views of top 10 videos')

plt.show()
likes = sns.jointplot(x ='views', y ='likes', data=videos_usa, kind='reg')

likes.annotate(stats.pearsonr)

comment_count = sns.jointplot(x ='views', y ='comment_count', data=videos_usa, kind='reg')

comment_count.annotate(stats.pearsonr)

dislikes = sns.jointplot(x ='views', y ='dislikes', data=videos_usa, kind='reg')

dislikes.annotate(stats.pearsonr)

plt.figure(figsize=(16, 9), dpi=200)

plt.show()