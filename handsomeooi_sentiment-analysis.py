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
# if pySpark not yet installed
!pip install pyspark
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import random
import os

from pyspark.sql import SparkSession 
from pyspark.ml  import Pipeline     
from pyspark.sql import SQLContext  
from pyspark.sql.functions import mean,col,split, col, regexp_extract, when, lit
spark = SparkSession.builder.appName('sentiment').getOrCreate()
file = '/kaggle/input/bitcoin-tweets-20160101-to-20190329/tweets.csv'
df = spark.read.option('delimiter',';').csv(file, inferSchema=True, header=True)
df.limit(10).toPandas()   # this shows pySpark reads the CSV wrongly :()
from textblob import TextBlob
csv_path = r'/kaggle/input/bitcoin-tweets-20160101-to-20190329/tweets.csv'
df = pd.read_csv(csv_path, sep=';', nrows=10000)                 # just 10000 rows for a start

df['timestamp'] = pd.to_datetime(df['timestamp'])                # convert to date/time

df.head()
df.shape
df.sort_values(by='timestamp', inplace=True)        # resort the DF based on date time (oldest=first)
df.reset_index(drop=True, inplace=True)
df.head()
tweets = df['text']    # the 'text' column, containing the tweets
def calc_sentiment(txt):
    blob = TextBlob(txt)
    return blob.sentiment.polarity, blob.sentiment.subjectivity

tweets_sentiments = tweets.apply(calc_sentiment)               # calc sentiment polarity & subjectivity, return in a Series of tuples
tweets_polarity = tweets_sentiments.apply(lambda x: x[0])      # new column of polarity
tweets_subjectivity = tweets_sentiments.apply(lambda x: x[1])  # new column of subjectivity

df['polarity'], df['subjectivity'] = tweets_polarity, tweets_subjectivity  # create the series

df.sample(10)        # display 10 random rows
import matplotlib.pyplot as plt
%matplotlib inline
df.polarity.loc[df.polarity != 0.0].hist(bins=50)   # histogram of polarity (excluding those with neutral sentiment)
non_neutral_df = df.loc[df.polarity != 0.0]
non_neutral_df.reset_index(drop=True, inplace=True)
non_neutral_df.head().text.iloc[0]
random_tweets = df.text.sample(100)
random_tweets.to_csv('./random_tweets.csv')
random_tweets

df.polarity.rolling(window=200).mean().plot(color='darksalmon', linewidth=1)
plt.title('Average Sentiment of Bitcoin Tweets')
plt.xlabel('Tweet Index')
plt.ylabel('Sentiment Polarity (-1 to 1)');
df.subjectivity.rolling(window=200).mean().plot(color='green', linewidth=1)
plt.title('Average Subjectivity of Bitcoin Tweets')
plt.xlabel('Tweet Index')
plt.ylabel('Sentiment Subjectivity (0 to 1)');
# 10 most positive tweets
most_positive = df.sort_values(by='polarity', ascending=False)['text'].head(10)
for i in range(len(most_positive)):
    print(most_positive.iloc[i])
    print('-' * 50)
# 10 most negative tweets
most_negative = df.sort_values(by='polarity', ascending=True)['text'].head(10)
for i in range(len(most_negative)):
    print(most_negative.iloc[i])
    print('-' * 50)
# 10 most liked tweets
most_like_tweets = df.sort_values(by='likes', ascending=False).head(10)
most_like_tweets['text'].apply(lambda x: print(x + '\n' + '-'*50))
# 10 most replied tweets
most_replied_tweets = df.sort_values(by='replies', ascending=False).head(10)
most_replied_tweets['text'].apply(lambda x: print(x + '\n' + '-'*50))
