import pandas as pd

import os
d_fake = pd.read_csv('../input/fake-news-data/fnn_politics_fake.csv')

headlines_fake = d_fake.drop(['id', 'news_url', 'tweet_ids'], axis=1).rename(columns={'title': 'headline'})

headlines_fake['fake'] = 1

headlines_fake.head()
d_real = pd.read_csv('../input/fake-news-data/fnn_politics_real.csv')

headlines_real = d_real.drop(['id', 'news_url', 'tweet_ids'], axis=1).rename(columns={'title': 'headline'})

headlines_real['fake'] = 0

headlines_real.head()
data1 = pd.concat([headlines_fake, headlines_real])

data1 = data1.sample(frac=1).reset_index(drop=True)

data1.head()
def read_data(d):

    """Each file has a headline as the first line, followed by some white space and then the article content.

    We need to exract the headline and the content of each file and store them in lists."""

    fnames = os.listdir(d)

    headlines, contents = [], []

    for fname in fnames:

        f = open(d + '/' + fname)

        text = f.readlines()

        f.close()



        if len(text) == 2:

            # One of the lines is missing

            if len(text[1]) <= 1:

                # There is no article content or headline

                continue

        elif len(text) >= 3:

            # More than one empty line encountered

            text[1] = text[-1]

        else:

            # Only one or zero lines is file

            continue

        

        headline, content = text[0][:-1].strip().rstrip(), text[1][:-1]

        headlines.append(headline)

        contents.append(content)

    

    return headlines, contents
fake_dir = '../input/fake-news-data/fnd_news_fake'

fake_headlines, fake_content = read_data(fake_dir)
fake_headlines = pd.DataFrame(fake_headlines, columns=['headline'])

fake_headlines['fake'] = 1

fake_headlines.head()
real_dir = '../input/fake-news-data/fnd_news_real'

real_headlines, real_content = read_data(real_dir)
real_headlines = pd.DataFrame(real_headlines, columns=['headline'])

real_headlines['fake'] = 0

real_headlines.head()
data2 = pd.concat([fake_headlines, real_headlines])

data2 = data2.sample(frac=1).reset_index(drop=True)

data2.head()
fnames = os.listdir('../input/nyt-comments')
million = pd.read_csv('../input/million-headlines/abcnews-date-text.csv', delimiter=',')

million = million.drop(['publish_date'], axis=1).rename(columns={'headline_text': 'headline'})

million.head()
all_news = pd.read_csv('../input/all-the-news/articles3.csv')

all_news = all_news.rename(columns={'title': 'headline'})

all_news = all_news['headline']

all_news.head()
examiner = pd.read_csv('../input/examine-the-examiner/examiner-date-text.csv')

examiner = examiner.drop(['publish_date'], axis=1).rename(columns={'headline_text': 'headline'})

examiner.tail()
reuters = pd.read_csv('../input/fake-news-data/reuters-newswire-2017.v5.csv')

reuters = reuters.drop(['publish_time'], axis=1).rename(columns={'headline_text': 'headline'})

reuters.head()
data = pd.concat([data1, data2])
from collections import Counter

import matplotlib.pyplot as plt



lengths = Counter(data['headline'].str.len())

keys, values = list(lengths.keys()), list(lengths.values())

plt.bar(keys, values)
from collections import defaultdict



lengths = defaultdict(int)

for h in data['headline']:

    lengths[len(h.split())] += 1



keys, values = list(lengths.keys()), list(lengths.values())

plt.bar(keys, values)