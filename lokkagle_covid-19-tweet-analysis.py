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
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('seaborn')
tweet_data = pd.read_csv('/kaggle/input/covid19-tweets/covid19_tweets.csv')
tweet_data.head()
tweet_data.info()
tweet_data.isna().sum()
tweet_data.isna().count()
def missing_data(data):
    """this function handles the missing data percenatge wise and unique values as well in tabular form"""
    total_count = tweet_data.isna().count()
    total_nulls = tweet_data.isnull().sum()
    percent_nulls = (tweet_data.isnull().sum()/tweet_data.isnull().count()*100)
    tb = pd.concat([ total_count, total_nulls, percent_nulls], axis=1, keys=[' total_count','Total nulls', 'null Percent'])
    types = []
    uni_vals = []
    for col in tweet_data.columns:
        dtype = str(tweet_data[col].dtype)
        uniques = tweet_data[col].nunique()
        types.append(dtype)
        uni_vals.append(uniques)
    tb['Types'] = types
    tb['Unique values'] = uni_vals
    return tb
missing_data(tweet_data)
def get_countplot(tweet_data):
    """this function handles the top 10 user specifications"""
    user_cols = ['user_name', 'user_location', 'source']
    for col in user_cols:
        tweet_data[col].value_counts().head(10).plot(kind = 'bar', figsize = (15,5))
        plt.show()

get_countplot(tweet_data)
tweet_data['text'].head()
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
ps = PorterStemmer()

for i in range(0, tweet_data.shape[0]):
    text = re.sub(pattern= '^[a-zA-Z]', repl= ' ', string= tweet_data['text'][i])
    text = re.sub(r"http\S+", "", text)
    text = text.lower()
    text = text.split()
    words = [ word for word in text if word not in set(stopwords.words('english'))]
    stem_words = [ps.stem(st_words) for st_words in words]
    final_words = ' '.join(stem_words)
    corpus.append(final_words)
corpus[:10]

from wordcloud import WordCloud
import matplotlib.pyplot as plt

wc = WordCloud(background_color='white', width=3000, height=2500).generate(str(corpus))
plt.figure(figsize=(10,10))
plt.title('most sound words')
plt.imshow(wc)
plt.axis('off')
plt.show()
tweet_data['date'].head() # it is object data type so convert them into datetime
tweet_data['new_date'] = pd.to_datetime(tweet_data['date'])
tweet_data['new_date'].head()
# create more columns on datetime so that we can analyse it
tweet_data['year'] = tweet_data['new_date'].dt.year
tweet_data['month'] = tweet_data['new_date'].dt.month
tweet_data['day'] = tweet_data['new_date'].dt.day
tweet_data['dayofweek'] = tweet_data['new_date'].dt.dayofweek
tweet_data['hour'] = tweet_data['new_date'].dt.hour
tweet_data['minute'] = tweet_data['new_date'].dt.minute
tweet_data['dayofyear'] = tweet_data['new_date'].dt.dayofyear
tweet_data['date_only'] = tweet_data['new_date'].dt.date
# consider only datetime parts columns
cols1 = ['text', 'year', 'month', 'day', 'dayofweek',
       'hour', 'minute', 'dayofyear', 'date_only']
tweet_data.groupby(['year', 'month'])['text'].count().plot(kind = 'bar', figsize = (15,5))
plt.show()
tweet_data.groupby( ['month', 'day'])['text'].count().plot(kind = 'bar', figsize = (15,5))
plt.show()
tweet_data.groupby( ['day'])['text'].count().plot(kind = 'bar', figsize = (15,5))
plt.show()
tweet_data.groupby( ['day', 'hour'])['text'].count().plot(kind = 'bar', figsize = (18,5))
plt.show()
