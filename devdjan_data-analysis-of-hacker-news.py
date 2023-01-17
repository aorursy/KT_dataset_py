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
# import some packages

from datetime import datetime

import matplotlib.pyplot as plt

import re

%matplotlib inline



# Read data in the data set

hack_news = pd.read_csv('../input/HN_posts_year_to_Sep_26_2016.csv')

hack_news.head()

#
hack_news[['title', 'url', 'num_points']].sort_values(by='num_points', ascending=False)[0:10]
hack_news['domain'] = hack_news['url'].str.extract('^http[s]*://([0-9a-z\-\.]*)/.*$', flags = re.IGNORECASE, expand = False)

hn_groupby = hack_news.groupby(by = 'domain')

hn_groupby['num_points'].count().sort_values(ascending = False)[0:25]
# Upvotes, who will be on first two places

# So we will be accessing variable called num_points



hn_groupby['num_points'].sum().sort_values(ascending=False)[0:20]
# Let's find .mean of 10+ posts

# It will be blog about programming language Rust

hn_groupby['num_points'].mean()[hn_groupby['num_points'].count() > 9].sort_values(ascending = False)[0:20]
# hack_news['hour'] = hack_news['created_at'].dt.hour

# hn_groupby = hack_news.groupby(by='hour')

# hn_groupby['num_points'].mean().sort_values(ascending=False)

# Top 15 users

hn_groupby = hack_news.groupby(by='author')

hn_groupby['num_points'].sum().sort_values(ascending=False)[0:15]
hn_groupby['num_points'].mean()[hn_groupby['num_points'].count() > 9].sort_values(ascending=False)[0:15]
import seaborn as sns

sns.set(color_codes = True)



plot = sns.lmplot('num_comments', 'num_points', hack_news, fit_reg = False)

plot.set(ylim = (0, 900), xlim = (0, 900))
#hack_news['created_at'] = pd.to_datetime(hack_news.created_at)

# hack_news.created_at.dt.hour

# hack_news.created_at.max() 

# Timestamp('2016-09-26 03:26:00')

# hack_news('Year') = hack_news.created_at.dt.Year

%matplotlib inline

hack_news.created_at.value_counts().sort_index().plot()
# Trying to print out the activity during the year

# by_year = hack_news.groupby("created_at").size()

# hack_news.set_index('created_at').groupby('D/M/Y/ H:M').size()

# hack_news.set_index('created_at').groupby(pd.TimeGrouper('D')).size()

#  hacker_news.set_index('created_ad').groupby(pd.TimeGrouper('D')).size()