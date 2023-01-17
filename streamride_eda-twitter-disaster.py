# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.


import matplotlib.pyplot as plt

%matplotlib inline
train_data = pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')

test_data = pd.read_csv('/kaggle/input/nlp-getting-started/test.csv')
train_data.head()
train_data.shape
train_data[~train_data['keyword'].isna()]
train_data['target'].value_counts()
from collections import Counter

import nltk

import re
print('Average length of Disaster tweet is ', train_data[train_data['target'] == 1]['text'].apply(lambda x: len(x)).mean())

print('Average length of Not Disaster tweet is ', train_data[train_data['target'] == 0]['text'].apply(lambda x: len(x)).mean())
ax = train_data[train_data['target'] == 1]['text'].apply(lambda x: len(x)).hist()

plt.title('Lenght of disaster tweets')
ax = train_data[train_data['target'] == 0]['text'].apply(lambda x: len(x)).hist()

plt.title('Lenght of not disaster tweets')
ttokenizer = nltk.TweetTokenizer()

ttokenizer.tokenize(train_data.iloc[0].text)

ttokenizer.tokenize(train_data.iloc[1].text)
import string

from nltk.corpus import stopwords



stop_words = set(stopwords.words('english') + ['...', '\x89'])

def clean_text(text):

    """

    1. Lower text

    2. Remove standard puntuation

    3. Remove stop words

    """

    text = str(text).lower()

    words = ttokenizer.tokenize(text)

    text = ' '.join([word for word in words if (word not in set(string.punctuation) and word not in stop_words)]) 

    return text
train_data['clean_text'] = train_data['text'].apply(lambda x: clean_text(x))
counter = Counter(" ".join(train_data["clean_text"]).split())

counter.most_common(100)
counter = Counter(" ".join(train_data[train_data['target'] == 1]["clean_text"]).split())

counter.most_common(100)
counter = Counter(" ".join(train_data[train_data['target'] == 0]["clean_text"]).split())

counter.most_common(100)
train_data[train_data['clean_text'].apply(lambda x: 'http' in x)].iloc[0]
train_data[(train_data['clean_text'].apply(lambda x: 'http' in x) & (train_data['target'] == 1))]
train_data[(train_data['clean_text'].apply(lambda x: 'http' in x) & (train_data['target'] == 0))]
def extract_hashtags(text):

    hashtags = re.findall(r"#(\w+)", text)

    if len(hashtags) > 0:

        return hashtags

    return None



train_data['hashtags'] = train_data['clean_text'].apply(extract_hashtags)
train_data.head(10)
## Collect all disaster hashtags

disaster_hts = " ".join([ht for hts in train_data[(train_data['target'] == 1) & (~train_data['hashtags'].isna())]["hashtags"] for ht in hts])

## Compute frequency

disater_freq = nltk.FreqDist(disaster_hts.split())



distaster_freq_df = pd.DataFrame({'count': list(disater_freq.values()), 'hashtag': list(disater_freq.keys())})

distaster_freq_df = distaster_freq_df.nlargest(columns="count", n = 15) 



## Visualize 



plt.figure(figsize=(17,7))

plt.barh(y=distaster_freq_df['hashtag'], width=distaster_freq_df['count'], color=['b', 'g', 'r', 'c', 'm', 'y', 'g'])

plt.title('Disaster top hashtags')

plt.xlabel('Count')

plt.ylabel('Hashtag')

plt.show()
notdisaster_hts = " ".join([ht for hts in train_data[(train_data['target'] == 0) & (~train_data['hashtags'].isna())]["hashtags"] for ht in hts])



notdisater_freq = nltk.FreqDist(notdisaster_hts.split())

notdistaster_freq_df = pd.DataFrame({'count': list(notdisater_freq.values()), 'hashtag': list(notdisater_freq.keys())})

notdistaster_freq_df = notdistaster_freq_df.nlargest(columns="count", n = 15) 

plt.figure(figsize=(17,7))

plt.barh(y=notdistaster_freq_df['hashtag'], width=notdistaster_freq_df['count'], color=['b', 'g', 'r', 'c', 'm', 'y', 'g'])

plt.title('Not disaster top hashtags')

plt.xlabel('Count')

plt.ylabel('Hashtag')

plt.show()