# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from google.cloud import bigquery # import bigquery api
import bq_helper 
#client = bigquery.Client()
h_news = bq_helper.BigQueryHelper(active_project = "bigquery-public-data" , dataset_name = "hacker_news")
h_news.list_tables()
h_news.head("full")
h_news.head("comments")
h_news.head("stories")
client = bigquery.Client()
query = """select * from `bigquery-public-data.hacker_news.stories`
WHERE REGEXP_CONTAINS(title, r"(b|B)itcoin") order by time_ts
"""
#h_news.estimate_query_size(query)
results_pd = h_news.query_to_pandas_safe(query)
results_pd.head()
import wordcloud
import matplotlib.pyplot as plt

words = ' '.join(results_pd.title).lower()
cloud = wordcloud.WordCloud(background_color='black',
                            max_font_size=200,
                            width=1600,
                            height=800,
                            max_words=500,
                            relative_scaling=.5).generate(words)
plt.figure(figsize=(20,10))
plt.axis('off')
plt.savefig('kaggle-hackernews.png')
plt.imshow(cloud);
query = """SELECT REGEXP_EXTRACT(url, '//([^/]*)/?') top_domains, COUNT(*)
            counts, COUNTIF(score > 40) scores
            FROM `bigquery-public-data.hacker_news.full`
            WHERE url !='' and EXTRACT(YEAR from TIMESTAMP) =2017
            GROUP BY 1 order BY 3 DESC LIMIT 10"""
#h_news.estimate_query_size(query)
h_news.query_to_pandas_safe(query)
query = """select EXTRACT(HOUR FROM TIMESTAMP_SECONDS(time-3600*5))  hour, COUNT(*) comments, 
            AVG(ranking) avg_ranking,  ROUND(100*COUNTIF(ranking=0)/COUNT(*),2) prob
            from `bigquery-public-data.hacker_news.comments` 
            where EXTRACT(YEAR FROM time_ts) = 2015
            GROUP BY 1
            ORDER BY 1"""
comments_per_hour = h_news.query_to_pandas_safe(query)
comments_per_hour.set_index('hour')['comments'].plot(kind = 'line')
import seaborn as sns

query = """
SELECT *
FROM `bigquery-public-data.hacker_news.stories` 
WHERE REGEXP_CONTAINS(title, r"( [Ii][Oo][tT] | [Ii]nternet [Oo][Ff] [Tt]hings) ") 
OR REGEXP_CONTAINS(title, r"(^[Ii][Oo][tT]|^[Ii]nternet [Oo][Ff] [Tt]hings) ") 
OR REGEXP_CONTAINS(title, r"( [Ii][Oo][tT].|[Ii]nternet [Oo][Ff] [Tt]hings).") 
OR REGEXP_CONTAINS(title, r"( [Ii][Oo][tT]$|[Ii]nternet [Oo][Ff] [Tt]hings)$") 
ORDER BY time
"""
iot_trend = h_news.query_to_pandas_safe(query)
iot_trend
iot_trend['date'] = pd.to_datetime(iot_trend.time_ts)
iot_trend['year'] = iot_trend.date.dt.year.astype('category')
sns.set(style="whitegrid", color_codes=True)
plt.figure(figsize=(12,8))
#ax = sns.countplot(x="year",palette=my_colors, data=df_iot)
sns.countplot(x="year", data=iot_trend);


def get_type(str):
    str_type = ''
    str = str.lower()
    str = str.replace('.', '') # Take care of space.
    str = str.replace(',', '') # Take care of Comma.
    s_split = str.split(' ')
    if (('microsoft' in s_split) | ('azure' in s_split)):
       str_type = 'Microsoft'
    if (('aws' in s_split) | ('amazon' in s_split)):
       str_type = 'Amazon'
    if (('cisco' in s_split)):
       str_type = 'Cisco'
    if (('ibm' in s_split) | ('bluemix' in s_split)):
       str_type = 'IBM'
    if (('google' in s_split) | ('gcp' in s_split)):
       str_type = 'Google'
    return str_type

iot_trend['Type'] = iot_trend.title.apply(get_type)
iot_trend_majors = iot_trend[iot_trend.Type != ""]
sns.set(style="whitegrid", color_codes=True)
plt.figure(figsize=(12,8))
sns.countplot(x="Type", data=iot_trend_majors).set(xlabel='Company', 
                                                                      ylabel='# of appearances');
iot_trend_majors['Mentions'] = 1
iot_trend_majors_grouped_years_type = iot_trend_majors['Mentions'].groupby([iot_trend_majors['year'], iot_trend_majors['Type']]).count()
iot_trend_majors_grouped_years_type = iot_trend_majors_grouped_years_type.reset_index()
plt.figure(figsize=(11,8))
sns.pointplot(x="year", y="Mentions", hue="Type", data=iot_trend_majors_grouped_years_type);