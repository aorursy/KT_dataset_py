# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from collections import Counter
df = pd.read_csv('../input/train.csv')
# random article overview
df.loc[200].to_dict()
# title
print("Title")
print("-"*80)
print(df.loc[200]['title'])
# summary
print("Summary")
print("-"*80)
print(df.loc[200]['summary'])
df.head(5)
print( "Number of artiles with flag = 1 (included in BBB): \t{:,}".format(df[df['flag']==1]['id'].count()))
print( "Number of artiles with flag = 0 (NOT included in BBB): \t{:,}".format(df[df['flag']==0]['id'].count()))
# distribution of 1s by newspaper website
df[df['flag']==1].groupby('website').size().sort_values(0, ascending=False).reset_index().rename(columns={0:"count"}).head(15)
# distribution of 0s by newspaper website
df[df['flag']==0].groupby('website').size().sort_values(0, ascending=False).reset_index().rename(columns={0:"count"}).head(15)
# distribution of articles by day of week
df.groupby('day_of_week').size().sort_values(0, ascending=False).reset_index().rename(columns={0:"count"})
df.groupby(['day_of_week', 'flag']).size().reset_index().rename(columns={0:"count"})
# checking the keywords
keywords = []
for article_keywords in df['keywords']:
    for item in article_keywords[1:-1].split(','):
        keywords.append(item)
        
keywords_counts = Counter(keywords)

df_keywords = pd.DataFrame.from_dict(keywords_counts, orient='index').reset_index().rename(columns={'index':'keyword', 0:'count'})
df_keywords[df_keywords['keyword'].str.len()>5].sort_values('count', ascending=False).head(30)