#importing necessary libraries

import json
import pandas as pd
import numpy as np
import collections, re

#NLP libraries
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud, STOPWORDS

#for visualization
import seaborn as sns
import matplotlib
from matplotlib import pyplot as plt
%matplotlib inline

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))
with open("../input/smsCorpus_en_2015.03.09_all.json") as f:
    data = json.load(f)

type(data)
listofDict = data['smsCorpus']['message']
len(listofDict)
listofDict[0]
fullData = pd.DataFrame(listofDict)
smsData = fullData[['@id','text']]
smsData = pd.DataFrame(smsData)
smsData.head()
smsData['word_count'] = smsData['text'].apply(lambda x: len(str(x).split(" ")))
smsData[['text','word_count']].head()
smsData.head()
def avg_word(sentence):
  words = sentence.split()
  return (sum(len(word) for word in words)/len(words))

smsData['avg_word'] = smsData['text'].apply(lambda x: avg_word(str(x)))
smsData[['text','avg_word']].head()

from nltk.corpus import stopwords
stop = stopwords.words('english')

smsData['stopwords'] = smsData['text'].apply(lambda x: len([x for x in str(x).split() if x in stop]))
smsData[['text','stopwords']].head()
smsData['text'] = smsData['text'].apply(lambda x: " ".join(str(x).lower() for x in str(x).split()))
smsData['text'].head()
smsData['upper'] = smsData['text'].apply(lambda x: len([x for x in str(x).split() if x.isupper()]))
smsData[['text','upper']].head()
smsData['text'] = smsData['text'].str.replace('[^\w\s]','')
smsData['text'].head()
from nltk.corpus import stopwords
stop = stopwords.words('english')
smsData['text'] = smsData['text'].apply(lambda x: " ".join(x for x in str(x).split() if x not in stop))
smsData['text'].head()
freq = pd.Series(' '.join(smsData['text']).split()).value_counts()[:10]
freq
freq = list(freq.index)
smsData['text'] = smsData['text'].apply(lambda x: " ".join(x for x in str(x).split() if x not in freq))
smsData['text'].head()
rare = pd.Series(' '.join(smsData['text']).split()).value_counts()[-10:]
rare
rare = list(rare.index)
smsData['text'] = smsData['text'].apply(lambda x: " ".join(x for x in str(x).split() if x not in freq))
smsData['text'].head()
from textblob import TextBlob
smsData['text'][:5].apply(lambda x: str(TextBlob(x).correct()))
from nltk.stem import PorterStemmer
st = PorterStemmer()
smsData['text'][:5].apply(lambda x: " ".join([st.stem(word) for word in str(x).split()]))
from textblob import Word
smsData['text'] = smsData['text'].apply(lambda x: " ".join([Word(word).lemmatize() for word in str(x).split()]))
smsData['text'].head()
TextBlob(smsData['text'][3]).ngrams(2)
tf1 = (smsData['text'][1:2]).apply(lambda x: pd.value_counts(str(x).split(" "))).sum(axis = 0).reset_index()
tf1.columns = ['words','tf']
tf1
for i,word in enumerate(tf1['words']):
  tf1.loc[i, 'idf'] = np.log(smsData.shape[0]/(len(smsData[smsData['text'].str.contains(word)])))

tf1.sort_values(by=['idf'],ascending=False)
tf1['tfidf'] = tf1['tf'] * tf1['idf']
tf1.sort_values(by=['idf'],ascending=False)
topvacab = tf1.sort_values(by='tfidf',ascending=False)
top_vacab = topvacab.head(20)
sns.barplot(x='tfidf',y='words', data=top_vacab)
top_vacab.plot(x ='words', kind='bar') 
from sklearn.feature_extraction.text import CountVectorizer
bow = CountVectorizer(max_features=1000, lowercase=True, ngram_range=(1,1),analyzer = "word")
BagOfWords = bow.fit_transform(smsData['text'])
BagOfWords
smsData['sentiment'] = smsData['text'].apply(lambda x: TextBlob(str(x)).sentiment[0] )
sentiment = smsData[['text','sentiment']]

sentiment.head()
pos_texts = [ text for index, text in enumerate(smsData['text']) if smsData['sentiment'][index] > 0]
neu_texts = [ text for index, text in enumerate(smsData['text']) if smsData['sentiment'][index] == 0]
neg_texts = [ text for index, text in enumerate(smsData['text']) if smsData['sentiment'][index] < 0]

possitive_percent = len(pos_texts)*100/len(smsData['text'])
neutral_percent = len(neu_texts)*100/len(smsData['text'])
negative_percent = len(neg_texts)*100/len(smsData['text'])
percent_values = [possitive_percent, neutral_percent, negative_percent]
labels = 'Possitive', 'Neutral', 'Negative'

plt.pie(percent_values, labels=labels, autopct='%3.3f')
k= (' '.join(pos_texts))

wordcloud = WordCloud(width = 1000, height = 500).generate(k)
plt.figure(figsize=(15,5))
plt.imshow(wordcloud)
plt.axis('off')
k= (' '.join(neu_texts))

wordcloud = WordCloud(width = 1000, height = 500).generate(k)
plt.figure(figsize=(15,5))
plt.imshow(wordcloud)
plt.axis('off')
k= (' '.join(neg_texts))

wordcloud = WordCloud(width = 1000, height = 500).generate(k)
plt.figure(figsize=(15,5))
plt.imshow(wordcloud)
plt.axis('off')
