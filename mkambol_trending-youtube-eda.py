import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import statsmodels.api as sm

import seaborn as sns

import json

import datetime as dt

sns.set()  # sets seaborn as default for %matplotlib



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

func = func = lambda dates: [dt.datetime.strptime(x, '%y.%d.%m') for x in dates]



usvid = pd.read_csv('/kaggle/input/youtube-new/USvideos.csv', parse_dates=['trending_date'],date_parser=func)

usvid.head()

usvid.info()
usvid
with open('/kaggle/input/youtube-new/US_category_id.json') as categories:

    categories_dict = json.load(categories)



cats_list = [{'id':int(item['id']), 'category':item['snippet']['title']}  for item in categories_dict['items']]

cats = pd.DataFrame(cats_list)

cats.info()

usvids_cats = pd.merge(usvid, cats, how='inner', left_on=['category_id'], right_on=['id'])

usvids_cats.head()
usvids_cats[usvids_cats['video_id']=='8mhTWqWlQzU']
print('number unique trending videos:  ' + str(len(usvids_cats['video_id'].unique())))



most_days_trending = usvids_cats[['title', 'trending_date']].groupby('title').count().sort_values(['trending_date'], ascending=False).head(10).index.tolist()

most_days_trending


metrics_most_trending = usvids_cats[usvids_cats['title'].isin(most_days_trending)][['title','trending_date', 'views', 'likes', 'dislikes']]



metrics_most_trending
# removing missing impossible-- it has odd data.  Views go up, then drop, then go up again.

metrics_most_trending = metrics_most_trending[metrics_most_trending['title'] != 'Mission: Impossible - Fallout (2018) - Official Trailer - Paramount Pictures']
from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()

plt.figure(figsize=(40,20))



lp = sns.lineplot(x='trending_date', y='views', hue='title', data=metrics_most_trending[['title', 'trending_date', 'views', 'likes']])

plt.setp(lp.get_legend().get_texts(), fontsize='22') # for legend text

plt.setp(lp.get_legend().get_title(), fontsize='32') # for legend title
