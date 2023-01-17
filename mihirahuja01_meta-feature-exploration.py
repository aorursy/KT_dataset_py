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
import matplotlib.pyplot as plt 
import seaborn as sns 
nltk.download('punkt')
train = pd.read_csv('/kaggle/input/training-meta-info-nlp-tweets/train_v3.csv')
test = pd.read_csv('/kaggle/input/nlp-getting-started/test.csv')
train['text'] = train['text'].astype(str)
test['text'] = test['text'].astype(str)

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

train
fig = px.histogram(train, x="Number_of_words", color="target", marginal="rug", # can be `box`, `violin`
                         hover_data=train.columns)
fig.show()

sns.set()
sns.set(rc={'figure.figsize':(10,8.27)})

sns.kdeplot(train[train['target']==0].Number_of_words, color="red",shade=True) 
sns.kdeplot(train[train['target']==1].Number_of_words, color="blue",shade=True ) 

plt.show()

fig = px.histogram(train, x="Number_of_Sentences", color="target", marginal="rug", # can be `box`, `violin`
                         hover_data=train.columns)
fig.show()

sns.set()
sns.set(rc={'figure.figsize':(10,8.27)})

sns.kdeplot(train[train['target']==0].Number_of_Sentences, color="red",shade=True) 
sns.kdeplot(train[train['target']==1].Number_of_Sentences, color="blue",shade=True ) 

plt.show()

fig = px.histogram(train, x="Number_of_Unique_Words", color="target", marginal="rug", # can be `box`, `violin`
                         hover_data=train.columns)
fig.show()

sns.set()
sns.set(rc={'figure.figsize':(10,8.27)})

sns.kdeplot(train[train['target']==0].Number_of_Unique_Words, color="red",shade=True) 
sns.kdeplot(train[train['target']==1].Number_of_Unique_Words, color="blue",shade=True ) 

plt.show()

fig = px.histogram(train, x="Number_of_Stop_Words", color="target", marginal="rug", # can be `box`, `violin`
                         hover_data=train.columns)
fig.show()

sns.set()
sns.set(rc={'figure.figsize':(10,8.27)})

sns.kdeplot(train[train['target']==0].Number_of_Stop_Words, color="red",shade=True) 
sns.kdeplot(train[train['target']==1].Number_of_Stop_Words, color="blue",shade=True ) 

plt.show()

def Number_of_Hashtage(tweet):
  return tweet.count('#')
train['Number_of_Hashtage'] = train['text'].apply(lambda x: Number_of_Hashtage(x))
test['Number_of_Hashtage'] = test['text'].apply(lambda x: Number_of_Hashtage(x))

def Number_of_Mentions(tweet):
  return tweet.count('@')
train['Number_of_Mentions'] = train['text'].apply(lambda x: Number_of_Mentions(x))
test['Number_of_Mentions'] = test['text'].apply(lambda x: Number_of_Mentions(x))

fig = px.histogram(train, x="Number_of_Hashtage", color="target", marginal="rug", # can be `box`, `violin`
                         hover_data=train.columns)
fig.show()

fig = px.histogram(train, x="Number_of_Mentions", color="target", marginal="rug", # can be `box`, `violin`
                         hover_data=train.columns)
fig.show()

def Average_Word_Length(tweet):
  
  words = tweet.split()
  try:
      average = sum(len(word) for word in words) / len(words)
  except:
        average = 0
  return average
train['Average_Word_Length'] = train['text'].apply(lambda x: Average_Word_Length(x))
test['Average_Word_Length'] = test['text'].apply(lambda x: Average_Word_Length(x))

fig = px.histogram(train, x="Average_Word_Length", color="target", marginal="rug", # can be `box`, `violin`
                         hover_data=train.columns)
fig.show()

sns.set()
sns.set(rc={'figure.figsize':(10,8.27)})

sns.distplot(train[train['target']==0].Average_Word_Length, color="red") 
sns.distplot(train[train['target']==1].Average_Word_Length, color="blue") 

plt.show()

