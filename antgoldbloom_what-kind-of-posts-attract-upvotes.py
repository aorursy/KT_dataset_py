#load in libraries

import pandas as pd

import re

%matplotlib inline
#read in the data set

df_hn = pd.read_csv('../input/HN_posts_year_to_Sep_26_2016.csv',parse_dates=['created_at'],index_col=[0])
df_hn[['title','url','num_points']].sort_values(by='num_points',ascending=False)[0:10]
df_hn['domain'] = df_hn['url'].str.extract('^http[s]*://([0-9a-z\-\.]*)/.*$',flags=re.IGNORECASE,expand=False)

df_groupby = df_hn.groupby(by='domain')

df_groupby['num_points'].count().sort_values(ascending=False)[0:20]
df_groupby['num_points'].sum().sort_values(ascending=False)[0:20]

#ideally I'd strip out the subdomains
df_groupby['num_points'].mean()[df_groupby['num_points'].count() > 9].sort_values(ascending=False)[0:20]
df_hn['hour'] = df_hn['created_at'].dt.hour

df_groupby = df_hn.groupby(by='hour')

df_groupby['num_points'].mean().sort_values(ascending=False)

#should really strip out outliers before doing analyzing impact of hour of day or day of week
df_hn['dayofweek'] = df_hn['created_at'].dt.dayofweek

df_groupby = df_hn.groupby(by='dayofweek')

df_groupby['num_points'].mean().sort_values(ascending=False)

#Monday is 0 and Sunday is 6
##top 20 users whose posts attract the most upvotes

df_groupby = df_hn.groupby(by='author')

df_groupby['num_points'].sum().sort_values(ascending=False)[0:20]
df_groupby['num_points'].mean()[df_groupby['num_points'].count() > 9].sort_values(ascending=False)[0:20]