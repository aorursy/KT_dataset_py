import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import os
from collections import Counter

print(os.listdir("../input"))

with open('../input/tweets.csv','r') as infile:
    tweets = pd.read_csv(infile)
tweets['text'].head()
def getHashtags(text):
    for string in text:
        if isinstance(string,str):
            for token in string.split(' '):
                if token.startswith('#'):
                    yield token


hashtags = getHashtags(tweets['text'])
hashtag_frequency = Counter(hashtags)
hashtag_frequency.most_common(100)
def getRetweets(text):
    for string in text:
        if isinstance(string,str):
            tokens = string.split(' ')
            if tokens[0] == 'RT' and tokens[1].startswith('@'):
                yield tokens[1]


retweets = getRetweets(tweets['text'])
retweet_frequency = Counter(retweets)
retweet_frequency.most_common(100)
def getMentions(text):
    for string in text:
        if isinstance(string,str):
            for token in string.split(' '):
                if token.startswith('@') and len(token)>1:
                    yield token.partition('\'')[0].partition(':')[0]


mentions = getMentions(tweets['text']) 
mention_frequency = Counter(mentions)- retweet_frequency
mention_frequency.most_common(100)
plt.figure(figsize=(24,16))

plt.subplot(1,3,1)
top50 = hashtag_frequency.most_common(50)
keys = [x[0] for x in top50]
vals = [x[1] for x in top50]
xpos = range(len(vals),0,-1)
plt.barh(xpos, vals)
plt.yticks(xpos,keys)
plt.title("Hashtags")

plt.subplot(1,3,2)
top50 = mention_frequency.most_common(50)
keys = [x[0] for x in top50]
vals = [x[1] for x in top50]
xpos = range(len(vals),0,-1)
plt.barh(xpos, vals)
plt.yticks(xpos,keys)
plt.title("Mentions")

plt.subplot(1,3,3)
top50 = retweet_frequency.most_common(50)
keys = [x[0] for x in top50]
vals = [x[1] for x in top50]
xpos = range(len(vals),0,-1)
plt.barh(xpos, vals)
plt.yticks(xpos,keys)
plt.title("Retweets")

plt.subplots_adjust(wspace=0.3)
plt.show()
