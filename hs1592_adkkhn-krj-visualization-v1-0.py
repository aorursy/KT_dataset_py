import numpy as np

import pandas as pd

import string

from collections import defaultdict,Counter

import matplotlib.pyplot as plt

plt.style.use('ggplot')

from nltk.corpus import stopwords

stop=set(stopwords.words('english'))

%matplotlib inline

from matplotlib import rcParams 

import seaborn as sns
train = pd.read_csv('../input/nlp-getting-started/train.csv')

test = pd.read_csv('../input/nlp-getting-started/test.csv')

train.head(100)
# train.describe()
train.isna().sum()
print('There are {} rows and {} columns in train'.format(train.shape[0],train.shape[1]))

print('There are {} rows and {} columns in test'.format(test.shape[0],test.shape[1]))
x = train.target.value_counts()

sns.barplot(x.index,x)

print(x)
fig, (ax1,ax2)=plt.subplots(1,2,figsize=(10,5))

tweet_len = train[train['target']==1]['text'].str.len()

ax1.hist(tweet_len, color='red')

ax1.set_title('disaster')

tweet_len = train[train['target']==0]['text'].str.len()

ax2.hist(tweet_len, color='blue')

ax2.set_title('Not disaster')

fig.suptitle('Characters in tweets')

plt.show()
fig,(ax1,ax2) = plt.subplots(1,2,figsize=(10,5))

tweet_len = train[train['target']==1]['text'].str.split().map(lambda x: len(x))

ax1.hist(tweet_len, color='red')

ax1.set_title('disaster')

tweet_len = train[train['target']==0]['text'].str.split().map(lambda x: len(x))

ax2.hist(tweet_len, color='blue')

ax2.set_title('Not disaster')

fig.suptitle('Words in a tweet')

plt.show()
fig,(ax1,ax2) = plt.subplots(1,2,figsize=(10,5))

word = train[train['target']==1]['text'].str.split().apply(lambda x: [len(i) for i in x])

sns.distplot(word.map(lambda x: np.mean(x)),ax=ax1,color='red')

ax1.set_title('disaster')

word = train[train['target']==0]['text'].str.split().apply(lambda x: [len(i) for i in x])

sns.distplot(word.map(lambda x: np.mean(x)),ax=ax2,color='blue')

ax1.set_title('Not disaster')

fig.suptitle('Average word length in each tweet')

plt.show()
def create_corpus(target):

    corpus=[]

    for x in train[train['target']==target]['text'].str.split():

        for i in x:

            corpus.append(i)

    return corpus
corpus = create_corpus(1)

dic = defaultdict(int)

for word in corpus:

    dic[word] += 1

        

top = sorted(dic.items(),key=lambda x: x[1],reverse=True)[:10]

# print(top)

x,y=zip(*top)

plt.bar(x,y)
corpus = create_corpus(0)



dic = defaultdict(int)

for word in corpus:

    dic[word]+=1



top=sorted(dic.items(), key=lambda x:x[1],reverse=True)[:10] 

    

x,y=zip(*top)

plt.bar(x,y,color='blue')
plt.figure(figsize=(10,5))

corpus = create_corpus(1)



dic = defaultdict(int)

special = string.punctuation

print(special)

for i in (corpus):

    if i in special:

        dic[i] += 1

# print(dic)

x,y = zip(*dic.items())

plt.bar(x,y)
plt.figure(figsize=(10,5))

corpus = create_corpus(0)



dic = defaultdict(int)



special = string.punctuation

for i in corpus:

    if i in special:

        dic[i] += 1

# print(dic)

x,y = zip(*dic.items())

plt.bar(x,y,color='blue')
corpus = create_corpus(1)

counter=Counter(corpus)

most=counter.most_common()

x=[]

y=[]

for word,count in most[:40]:

    if (word not in stop) :

        x.append(word)

        y.append(count)

sns.barplot(x=y,y=x)
corpus = create_corpus(0)

counter=Counter(corpus)

most=counter.most_common()

x=[]

y=[]

for word,count in most[:40]:

    if (word not in stop) :

        x.append(word)

        y.append(count)

sns.barplot(x=y,y=x)
set(train['keyword'])
df = train[['keyword','text']]

df.dropna(subset=['keyword'])