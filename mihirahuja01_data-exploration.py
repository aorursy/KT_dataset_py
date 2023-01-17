import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import pandas as pd
import numpy as np
import re
import plotly.express as px
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
stopwords = set(STOPWORDS)
import collections
import nltk
from nltk.tokenize import sent_tokenize,word_tokenize

nltk.download('punkt')
train = pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')
test = pd.read_csv('/kaggle/input/nlp-getting-started/test.csv')
train.head()
train.describe()
def cleanhtml(raw_html):
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, '', raw_html)
    return cleantext

def removeurl(raw_text):
    clean_text = re.sub(r'^https?:\/\/.*[\r\n]*', '', raw_text, flags=re.MULTILINE)
    return clean_text



train['text'] = train['text'].apply(lambda x: cleanhtml(x))
test['text'] = test['text'].apply(lambda x: cleanhtml(x))

train['text'] = train['text'].apply(lambda x:removeurl(x))
test['text'] = test['text'].apply(lambda x: removeurl(x))
train['text'] = train['text'].map(lambda x: re.sub(r'\W+', ' ', x))
test['text'] = test['text'].map(lambda x: re.sub(r'\W+', ' ', x))

train
fig = px.bar(train.groupby('target')['target'].count().reset_index(name='count'), x='target', y='count',color='target')
fig.show()
data = train.groupby('location')['location'].count().reset_index(name='count').sort_values(by='count',ascending=False)
data = data.head(20)
fig = px.bar(data, x='location', y='count',color='location')
fig.show()
data = train.groupby(['location','target'])['location'].count().reset_index(name='count').sort_values(by='count',ascending=False)
data = data.head(20)
fig = px.bar(data, x="location", y='count',color='target')
fig.show()

def show_wordcloud(data, title = None):
    wordcloud = WordCloud(
        background_color='white',
        stopwords=stopwords,
        max_words=200,
        max_font_size=40, 
        scale=3,
        random_state=1 # chosen at random by flipping a coin; it was heads
    ).generate(str(data))

    fig = plt.figure(1, figsize=(12, 12))
    plt.axis('off')
    if title: 
        fig.suptitle(title, fontsize=20)
        fig.subplots_adjust(top=2.3)

    plt.imshow(wordcloud)
    plt.show()

show_wordcloud(train[train['target']==1]['text'])
show_wordcloud(train[train['target']==0]['text'])
data = train[train['target']==1].text.str.split(expand=True).stack().value_counts().reset_index(name='count')
data
fig = px.bar(data.head(50), x="count", y="index", orientation='h',color_discrete_sequence=px.colors.qualitative.Light24)
fig.show()
data = train[train['target']==0].text.str.split(expand=True).stack().value_counts().reset_index(name='count')
fig = px.bar(data.head(50), x="count", y="index", orientation='h',color_discrete_sequence=px.colors.qualitative.Light24)
fig.show()
# counts = collections.Counter()
# for sent in train[train['target']==1]['text']: 
#         words = nltk.word_tokenize(sent)
#         counts.update(nltk.bigrams(words))
# data = pd.DataFrame.from_dict(counts, orient='index').reset_index()
# data = data.sort_values(by=0,ascending=False)
# data.columns = ['Bi-gram','Count']
# data['Bi-gram'] = data['Bi-gram'].astype(str)
# data['Count'] = data['Count'].astype(int)
# fig = px.bar(data.head(50), x="Count", y="Bi-gram", orientation='h',color_discrete_sequence=px.colors.qualitative.Light24)
# fig.show()
# counts = collections.Counter()
# for sent in train[train['target']==0]['text']:
#     words = nltk.word_tokenize(sent)
#     counts.update(nltk.bigrams(words))
# data = pd.DataFrame.from_dict(counts, orient='index').reset_index()
# data = data.sort_values(by=0,ascending=False)
# data.columns = ['Bi-gram','Count']
# data['Bi-gram'] = data['Bi-gram'].astype(str)
# data['Count'] = data['Count'].astype(int)
# fig = px.bar(data.head(70), x="Count", y="Bi-gram", orientation='h',color_discrete_sequence=px.colors.qualitative.Light24)
# fig.show()
#Creating Meta Features
train = pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')
test = pd.read_csv('/kaggle/input/nlp-getting-started/test.csv')
train['text'] = train['text'].apply(lambda x: cleanhtml(x))
test['text'] = test['text'].apply(lambda x: cleanhtml(x))

train['text'] = train['text'].apply(lambda x:removeurl(x))
test['text'] = test['text'].apply(lambda x: removeurl(x))
train['text'] = train['text'].astype(str)
test['text'] = test['text'].astype(str)
def number_of_words(tweet):
  return len(word_tokenize(tweet))
#number_of_words('Our Deeds are')
train['Number_of_words'] = train['text'].apply(lambda x: number_of_words(x))
test['Number_of_words'] = test['text'].apply(lambda x: number_of_words(x))

def number_of_sentences(tweet):

   return len(sent_tokenize(tweet))

train['Number_of_Sentences'] = train['text'].apply(lambda x: number_of_sentences(x))
test['Number_of_Sentences'] = test['text'].apply(lambda x: number_of_sentences(x))

def Number_of_Unique_Words(tweet):
  return len(set(tweet.split()))
train['Number_of_Unique_Words'] = train['text'].apply(lambda x: Number_of_Unique_Words(x))
test['Number_of_Unique_Words'] = test['text'].apply(lambda x: Number_of_Unique_Words(x))

def Number_of_Stop_Words(tweet):
  word_tokens = word_tokenize(tweet) #splitta i pezzi

  stopwords_x = [w for w in word_tokens if w in STOPWORDS]

  return len(stopwords_x)
train['Number_of_Stop_Words'] = train['text'].apply(lambda x: Number_of_Stop_Words(x))
test['Number_of_Stop_Words'] = test['text'].apply(lambda x: Number_of_Stop_Words(x))

train
test
