# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import json

import matplotlib.pyplot as plt # plotting

import seaborn as sns

import collections

import PIL

from wordcloud import WordCloud

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# H1 What is The Political Ad Collector ?**Bold**It is a tool you add to your Web browser. It copies the ads you see on Facebook, so anyone, on any part of the political spectrum, can see them.More detailed info at - https://www.theglobeandmail.com/political-ad-collector/

    
data = pd.read_csv("../input/political-advertisements-from-facebook/fbpac-ads-en-US.csv")

data.head(3)
data.info()
data[['title','impressions']].groupby('title',sort=False)['impressions'].size().sort_values(ascending=False)[:50]

data[['advertiser','political']].sort_values(by='political',ascending=False)[:50]
adtar = data[['advertiser','political','targets']].sort_values(by='political',ascending=False)[:50]
adtar['targeting_segment']= [json.loads(x) for x in adtar['targets']]
most_targetted_age_segment=[y.get('segment') for x in adtar['targeting_segment'] for y in x if y.get('segment') != None ]

most_common = [item for item in collections.Counter(most_targetted_age_segment).most_common()]

print(dict(most_common))
import wordcloud



wordcloud = wordcloud.WordCloud(max_font_size=50, max_words=100, background_color="white").generate_from_frequencies(dict(most_common))

fig = plt.figure(figsize=[10,10])

plt.imshow(wordcloud,interpolation="bilinear")

plt.axis('off')

plt.show()
data[['advertiser','not_political','targets']].sort_values(by='not_political',ascending=False)[:10]
advertisers = data['advertiser'].value_counts()[:10]

plt.figure(figsize=(10,5))

sns.barplot(x=advertisers.index.str.replace(' ', '\n'),y=advertisers.values,alpha=0.8)

plt.title(' Top 10 Advertisers on Facebook')

plt.ylabel('Number of times an account posted the Ad', fontsize=10)

plt.xlabel('advertisers', fontsize=10)

plt.xticks(rotation=308)

plt.show()
paid=data['paid_for_by'].value_counts()[:10]

plt.figure(figsize=(10,5))

sns.barplot(x=paid.index.str.replace(' ', '\n'),y=paid.values,alpha=0.8)

plt.title(' Top 10 Advertisers on Facebook')

plt.ylabel('Number of times an account posted the Ad', fontsize=10)

plt.xlabel('advertisers', fontsize=10)

plt.xticks(rotation=45)

plt.show()