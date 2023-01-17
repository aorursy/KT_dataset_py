# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
chunks = pd.read_json('../input/yelp-dataset/yelp_academic_dataset_review.json', lines=True, orient='records', chunksize=500)
reviews = []
for chunk in chunks:
    reviews.append(chunk)
reviews_df = pd.concat(reviews)
reviews_df.head()
del reviews
reviews_df = reviews_df[['business_id', 'stars', 'useful', 'funny', 'cool', 'text', 'date']]
reviews_df.head()
reviews_df.shape
reviews_df[['business_id', 'text']].groupby('business_id').count().sort_values('text', ascending=False)
chunks = pd.read_json('../input/yelp-dataset/yelp_academic_dataset_business.json', lines=True, orient='records', chunksize=500)

business = []
for chunk in chunks:
    business.append(chunk)
    
business_df = pd.concat(business)
business_df.head()
del business
business_df = business_df[['business_id', 'name', 'state', 'latitude', 'longitude', 'categories']]
reviews_df.head()
business_df.head()
review_business_df = reviews_df.merge(business_df, on='business_id', how='left')
review_business_df.head()
review_business_df.dropna()
del reviews_df
del business_df
