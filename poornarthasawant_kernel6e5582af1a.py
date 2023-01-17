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
df = pd.read_csv('/kaggle/input/coronavirus-covid19-tweets-late-april/2020-04-19 Coronavirus Tweets.CSV')
df
df.columns
df = df[df['country_code'].notna()]
df
df = df.reset_index()
india = df['country_code'] == 'IN'
data = df[india]
data
count_users = data.groupby('place_full_name').count()['user_id'].sort_values(ascending=False)
total_users = count_users.sum()
normalized_user_count = count_users/total_users
normalized_user_count.sort_values(ascending=False)
data = data[data['lang'] == 'en']
usable = pd.DataFrame(data['text'])
usable
# Clean the data:
import re
moji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           "]+", flags=re.UNICODE)
def cleanData(text):
    text = re.sub(r'@[A-Za-z0-9]+', '', text)
    text = re.sub(r'#[A-Za-z0-9]+', '', text)
    text = re.sub(r'RT[\S]+', '', text)
    text = re.sub(r'https?:/\/\S+', '', text)
    text = re.sub(moji_pattern, '', text)
    return text
    
usable['text'] = usable['text'].apply(cleanData)
usable
from textblob import TextBlob
sent = []
for sentence in usable['text']:
    blob = TextBlob(sentence)
    sent.append(blob.sentiment.polarity)
usable['sentiment'] = sent
usable
p = len([x for x in usable['sentiment'] if x>0])
p
n = len([x for x in usable['sentiment'] if x<0])
n
usable.shape[0] - (p + n)
tweets_data = pd.DataFrame()
dates = []
positive = []
negative = []
for i in range(1, 30):
    if i >= 16:
        df = pd.read_csv('/kaggle/input/coronavirus-covid19-tweets-late-april/2020-04-'+ str(i) + ' Coronavirus Tweets.CSV')
    else:
        if i < 10:
            df = pd.read_csv('/kaggle/input/coronavirus-covid19-tweets-early-april/2020-04-0'+ str(i) + ' Coronavirus Tweets.CSV')
        else:
            df = pd.read_csv('/kaggle/input/coronavirus-covid19-tweets-early-april/2020-04-'+ str(i) + ' Coronavirus Tweets.CSV')
    date = '2020-04-'+ str(i)
    df = df[df['country_code'].notna()]
    df = df.reset_index()
    india = df['country_code'] == 'IN'
    data = df[india]
    data = data[data['lang'] == 'en']
    usable = pd.DataFrame(data['text'])
    moji_pattern = re.compile("["
            u"\U0001F600-\U0001F64F"  # emoticons
            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
            u"\U0001F680-\U0001F6FF"  # transport & map symbols
            u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               "]+", flags=re.UNICODE)
    def cleanData(text):
        text = re.sub(r'@[A-Za-z0-9]+', '', text)
        text = re.sub(r'#[A-Za-z0-9]+', '', text)
        text = re.sub(r'RT[\S]+', '', text)
        text = re.sub(r'https?:/\/\S+', '', text)
        text = re.sub(moji_pattern, '', text)
        return text

    usable['text'] = usable['text'].apply(cleanData)
    sent = []
    for sentence in usable['text']:
        blob = TextBlob(sentence)
        sent.append(blob.sentiment.polarity)
    dates.append(date)
    usable['sentiment'] = sent
    positive.append(len([x for x in usable['sentiment'] if x>0]) / len(usable['sentiment']))
    negative.append(len([x for x in usable['sentiment'] if x<0]) / len(usable['sentiment']))
tweets_data['Date'] = dates
tweets_data['Positive'] = positive
tweets_data['Negative'] = negative
tweets_data

