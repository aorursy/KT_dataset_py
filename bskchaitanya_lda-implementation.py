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
news = pd.read_csv('/kaggle/input/cbc-news-coronavirus-articles-march-26/news.csv')
news.head()
news.columns
cbc_news1 = news.drop(columns = 'Unnamed: 0')
cbc_news1.columns
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_df = 0.9,min_df=2,lowercase=True,stop_words='english')
news = cv.fit_transform(cbc_news1['text'])
from sklearn.decomposition import LatentDirichletAllocation
LDA = LatentDirichletAllocation(n_components = 30,random_state=42)
LDA.fit(news)
print(len(cv.get_feature_names()))
print('\n')
print(LDA.components_)
single_topic = LDA.components_[0]
single_topic.argsort()
top_15_words = single_topic.argsort()[-15:]
top_15_words
for index in top_ten_words:
    print(cv.get_feature_names()[index])
for index,topic in enumerate(LDA.components_):
    print(f"The top 15 Words for topic #{index}")
    print([cv.get_feature_names()[index] for index in topic.argsort()[-15:]])
    print('\n')
    print('\n')
topic_results = LDA.transform(news)
topic_results[0].round(2)
topic_results[0].argmax()
cbc_news1['topic'] = topic_results.argmax(axis = 1)
cbc_news1.head()
