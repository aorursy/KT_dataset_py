# Disable warnings in Anaconda
import warnings
warnings.simplefilter('ignore')

from google.cloud import bigquery
from ggplot import *
from fbprophet import Prophet
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import wordcloud
import seaborn as sns

from pylab import rcParams
rcParams['figure.figsize'] = 8, 8

pd.options.display.float_format = '{:,.2f}'.format
%matplotlib inline
client = bigquery.Client()
query = """
#standard-sql
SELECT
  domain, count_dom, week_year, COUNT(*) posts
FROM (
  SELECT
    week_year, domain, COUNT(*) OVER(PARTITION BY domain) count_dom
  FROM (
    SELECT
      TIMESTAMP_TRUNC(timestamp, WEEK) week_year,
      REGEXP_EXTRACT(url, '//([^/]*)/?') domain
    FROM
      `bigquery-public-data.hacker_news.full`
    WHERE
      url!='' AND EXTRACT(YEAR FROM timestamp) IN (2017, 2018)) )
WHERE
  count_dom> 100
GROUP BY
  1, 2, 3
ORDER BY
  2 DESC,3
"""
query_job = client.query(query)
iterator = query_job.result(timeout=30)
rows = list(iterator)

top_domains_df = pd.DataFrame(data=[list(x.values()) for x in rows], columns=list(rows[0].keys()))
top_domains_df.head(5)
print("Shape of DataFrame: {0}\nNumber of Unique domains: {1}".format(top_domains_df.shape, 
                                                          len(top_domains_df.domain.unique())))
top_top_domains_df = top_domains_df[(top_domains_df.count_dom>3000) & 
                                        (top_domains_df.week_year < pd.datetime(2018,2,11))  ]

g = ggplot(top_top_domains_df, aes(x = 'week_year', y = 'posts', color = 'domain')) +\
geom_line(size  = 2) + facet_wrap('domain'); 

t = theme_gray()
t._rcParams['xtick.labelsize'] = 5
g + t
query = """
#standardSQL
SELECT DATE(timestamp) day, COUNT(*) posts FROM 
`bigquery-public-data.hacker_news.full`
WHERE EXTRACT(YEAR FROM timestamp) IN (2016, 2017, 2018)
GROUP BY 1
ORDER BY 1
"""
query_job = client.query(query)
iterator = query_job.result(timeout=30)
rows = list(iterator)

daily_posts_df = pd.DataFrame(data=[list(x.values()) for x in rows], columns=list(rows[0].keys()))
daily_posts_df.head(5)
daily_posts_df.day = pd.to_datetime(daily_posts_df.day, format='%Y-%m-%d')
daily_posts_df_short = daily_posts_df[daily_posts_df.day.dt.year > 2016]
daily_posts_df_short['posts'] = np.log(daily_posts_df_short['posts'])
daily_posts_df_short.columns = ['ds', 'y']

# https://facebook.github.io/prophet/docs/quick_start.html#python-api
m = Prophet()
m.fit(daily_posts_df_short);
# Predict posts pattern for next 6 months time
future = m.make_future_dataframe(periods=180)

forecast = m.predict(future)
m.plot(forecast); # logarithm of number of posts
m.plot_components(forecast);
# Here we consider posts with score > 20

query = """
SELECT REGEXP_EXTRACT(url, '//([^/]*)/?') domain, COUNT(*) n_posts, COUNTIF(score>20) n_posts_20
FROM `bigquery-public-data.hacker_news.full`
WHERE url!='' AND EXTRACT(YEAR FROM timestamp)=2017
GROUP BY 1 ORDER BY 3 DESC LIMIT 100
"""
query_job = client.query(query)
iterator = query_job.result(timeout=30)
rows = list(iterator)

fifty_score_df = pd.DataFrame(data=[list(x.values()) for x in rows], columns=list(rows[0].keys()))
fifty_score_df.head(5)
temp_df = fifty_score_df[fifty_score_df['n_posts'] > 2000]
domain_list = temp_df.sort_values(['n_posts'], ascending=[0]).domain.values

fifty_score_df_melt = pd.melt(temp_df, 
                              id_vars=['domain'], 
                              value_vars=['n_posts', 'n_posts_20'])
#fifty_score_df_melt.head(5)
rcParams['figure.figsize'] = 12, 8
ax = sns.barplot(x="domain", y="value", hue="variable", data=fifty_score_df_melt,
                order = domain_list)
for item in ax.get_xticklabels():
    item.set_rotation(60)    
ax.set(xlabel = "Domain", ylabel = "(1) # of Posts (2)# of Posts with > 20 votes");    
query = query = """
SELECT domain, score FROM (
    SELECT REGEXP_EXTRACT(url, '//([^/]*)/?') as domain, score
    FROM `bigquery-public-data.hacker_news.full` 
    WHERE EXTRACT(YEAR FROM timestamp)=2017)
WHERE domain in ('medium.com', 'www.nytimes.com', 'github.com', 'www.bloomberg.com') 
"""
query_job = client.query(query)
iterator = query_job.result(timeout=30)
rows = list(iterator)

top_domains_hist_df = pd.DataFrame(data=[list(x.values()) for x in rows], columns=list(rows[0].keys()))
#top_domains_hist_df.head(5)
g = top_domains_hist_df[top_domains_hist_df.domain == 'github.com']['score']
n = top_domains_hist_df[top_domains_hist_df.domain == 'www.nytimes.com']['score']
m = top_domains_hist_df[top_domains_hist_df.domain == 'medium.com']['score']
b = top_domains_hist_df[top_domains_hist_df.domain == 'www.bloomberg.com']['score']

rcParams['figure.figsize'] = 12, 8
sns.kdeplot(m.rename('Medium'), shade = True)
sns.kdeplot(g.rename('GitHub'), shade = True)
sns.kdeplot(n.rename('NYTimes'), shade = True)
sns.kdeplot(b.rename('Bloomberg'), shade = True)

plt.xlabel('Score received by the post');
plt.ylabel('Kernel density plot of posts\' scores in 2017');
top_domains_hist_df.groupby(['domain'])['score'].agg([np.mean, np.median])
query = """
SELECT `by` author, COUNT(DISTINCT domain) n_domains, SUM(score) total_score, AVG(score) avg_score, COUNT(*) AS n_posts FROM (
    SELECT `by`,REGEXP_EXTRACT(url, '//([^/]*)/?') as domain, score
    FROM `bigquery-public-data.hacker_news.full` 
    WHERE EXTRACT(YEAR FROM timestamp)=2017 AND `by` !='' AND url !='') 
GROUP BY 1 ORDER BY 5 DESC
LIMIT 5000
"""
query_job = client.query(query)
iterator = query_job.result(timeout=30)
rows = list(iterator)

top_users_df = pd.DataFrame(data=[list(x.values()) for x in rows], columns=list(rows[0].keys()))
#top_users_df.head(5)
temp_df = top_users_df.sort_values(['n_posts'], ascending =[0]).head(10)
temp_df
rcParams['figure.figsize'] = 12, 8
ax = sns.barplot(x="n_posts", y="author", data=temp_df)
ax.set(xlabel = "Number of posts made by author in 2017", ylabel = "Author of post");  
temp_df = top_users_df.sort_values(['avg_score'], ascending =[0]).head(10)
temp_df
ax = sns.barplot(x="avg_score", y="author", data=temp_df)
ax.set(xlabel = "Average score per post", ylabel = "Author of post"); 
# Most diverse posters
temp_df = top_users_df.sort_values(['n_domains'], ascending =[0]).head(10)
temp_df
ax = sns.barplot(x="n_domains", y="author", data=temp_df)
ax.set(xlabel = "Number of unique-domains posted by author, 2017", ylabel = "Author of post"); 
# Taking an arbitrary average of 100 score per post
temp_df = top_users_df[top_users_df.avg_score > 100].sort_values(['n_domains'], ascending =[0]).head(10)
temp_df
ax = sns.barplot(x="n_domains", y="author", data=temp_df)
ax.set(xlabel = "Number of unique-domains posted by top-scoring authors, 2017", ylabel = "Author of post"); 