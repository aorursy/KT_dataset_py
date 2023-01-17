import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline



import datetime as dt



pd.set_option('display.max_colwidth',1000)
df = pd.read_csv('../input/train.csv')
# Stack the questions from wide format to tall format

dfq = pd.concat(

    (df[['qid1','question1','is_duplicate']].rename(columns={'qid1': 'qid', 'question1': 'question'}), 

     df[['qid2','question2','is_duplicate']].rename(columns={'qid2': 'qid', 'question2': 'question'}))

).set_index('qid').sort_index()



dfq = dfq.dropna(subset=['question'])
dfq.head(1)
import string

punc = set(string.punctuation)



def strip_punctuation(x):

    x = ''.join([char for char in x if char not in punc])

    return x
dfq['words'] = dfq.question.apply(lambda ques: set(strip_punctuation(word.lower()) for word in ques.split()))
def find_phrase(phrase):

    words = set(w.lower() for w in phrase.split())

    exists = dfq.words.apply(lambda x: words.issubset(x))

    return exists
# Try searching some phrase

mask = find_phrase('resolution 2017')

dfq[mask]['question'].sample(10)
events = {'uri attack': dt.date(2016,9,18),

          'hurricane matthew': dt.date(2016,9,28),

          'brexit': dt.date(2016, 6, 23),

          'nba finals 2016': dt.date(2016, 6, 19),

          'nba finals 2015': dt.date(2015, 6, 16),

          'resolution 2015': dt.date(2015, 1, 1),

          'resolution 2016': dt.date(2016, 1, 1),

          'resolution 2017': dt.date(2017, 1, 1),

         }
dates_ids = {}

for evt, date in sorted(events.items(), key=lambda x: x[1]):

    mask = find_phrase(evt)

    if mask.max()>0:

        dates_ids.update({date: dfq[mask>0].index.values})

    print(evt, date, ' found', mask.sum())
colors = list('bgrcmykb')

colors.reverse()

for date, qid in sorted(dates_ids.items(), key=lambda x: x[0]):

    plt.scatter([date]*len(qid), qid, c=colors.pop(), s=50, alpha=.25, )

plt.ylim([0, plt.ylim()[1]])

plt.xticks(rotation=70)

plt.ylabel('qid')

plt.title('date of event vs question id - is there a trendline?')