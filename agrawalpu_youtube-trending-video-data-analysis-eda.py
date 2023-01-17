# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
us_df = pd.read_csv('../input/USvideos.csv', index_col='video_id')
us_df.shape
us_df.info()
import json

id_category_dict = {}

with open('../input/FR_category_id.json', 'r') as f:
    content = json.load(f)
    for item in content['items']:
        id_category_dict[item['id']] = item['snippet']['title']

id_category_dict
us_df.head()
us_df[['trending_date', 'publish_time']].head()

us_df['trending_date'] = pd.to_datetime(us_df['trending_date'], format='%y.%d.%m')
us_df['trending_date'].head()
us_df['publish_time'] = pd.to_datetime(us_df['publish_time'], format='%Y-%m-%dT%H:%M:%S.%fZ')
us_df['publish_time'].head()
#DataFrame.insert(loc, column, value, allow_duplicates=False)[source]
us_df.insert(3, 'publish_date', us_df['publish_time'].dt.date)
us_df['publish_time'] = us_df['publish_time'].dt.time
us_df[['publish_time', 'publish_date', 'trending_date']].head()
us_df.head(2)

for column in ['category_id']:
    us_df[column] = us_df[column].astype(str)

us_df.insert(3, 'category', us_df['category_id'].map(id_category_dict))
us_df[['category_id', 'category']].head()
us_df.head(2)
%matplotlib inline
us_df.category.value_counts().plot(kind='bar', title='Most trending category')
