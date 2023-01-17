# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from collections import Counter
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
import string
import sys
import re

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
import cufflinks as cf
cf.go_offline()
cf.set_config_file(offline=False, world_readable=True)
df=pd.read_csv('/kaggle/input/file1.csv')
df.head(10)
df['intent'].value_counts()
from nltk import word_tokenize 
from nltk.util import ngrams
def abc():
    x=(" ".join(df["query"]).split())
    return x   
count1 = Counter(" ".join(df["query"]).split()).most_common(20)
df1 = pd.DataFrame.from_dict(count1)
df1 = df1.rename(columns={0: "words", 1 : "count"})
fig = plt.figure()
ax = fig.add_subplot(111)
df1.plot.bar(ax=ax, legend = False)
xticks = np.arange(len(df1["words"]))
ax.set_xticks(xticks)
ax.set_xticklabels(df1["words"])
ax.set_ylabel('Number of occurences')
plt.show()
def get_top_n_words(corpus, n=None):
    vec = CountVectorizer(stop_words = 'english').fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]
common_words = get_top_n_words(df['query'], 20)
for word, freq in common_words:
    print(word, freq)
df2 = pd.DataFrame(common_words, columns = ['ReviewText' , 'count'])
df2.groupby('ReviewText').sum()['count'].sort_values(ascending=False).iplot(
    kind='bar', yTitle='Count', linecolor='black', title='Top 20 words in review after removing stop words')
wc_height, wc_width = (512, 1024)
wc_bckp_color = 'white'
wc_max_words = 400
wc_max_font_size = 60
random_state = 42
wc_figsize = (12, 10)
spam_df = df.loc[df['intent'] == 'cancel']

# Creating wordcloud for spam
spam_wc = WordCloud(
    height=wc_height, width=wc_width, background_color=wc_bckp_color,
    max_words=wc_max_words, max_font_size=wc_max_font_size,
    random_state=random_state
).generate(str(spam_df['query']))

# Display the wordcloud
fig = plt.figure(figsize=wc_figsize)
plt.imshow(spam_wc)
plt.axis('off')
plt.show()
spam_df = df.loc[df['intent'] == 'book']

# Creating wordcloud for spam
spam_wc = WordCloud(
    height=wc_height, width=wc_width, background_color=wc_bckp_color,
    max_words=wc_max_words, max_font_size=wc_max_font_size,
    random_state=random_state
).generate(str(spam_df['query']))

# Display the wordcloud
fig = plt.figure(figsize=wc_figsize)
plt.imshow(spam_wc)
plt.axis('off')
plt.show()
spam_df = df.loc[df['intent'] == 'status']

# Creating wordcloud for spam
spam_wc = WordCloud(
    height=wc_height, width=wc_width, background_color=wc_bckp_color,
    max_words=wc_max_words, max_font_size=wc_max_font_size,
    random_state=random_state
).generate(str(spam_df['query']))

# Display the wordcloud
fig = plt.figure(figsize=wc_figsize)
plt.imshow(spam_wc)
plt.axis('off')
plt.show()
df.loc[df["intent"] == "book", "intent", ] = 0
df.loc[df["intent"] == "cancel", "intent", ] = 1
df.loc[df["intent"] == "check-in", "intent", ] = 2
df.loc[df["intent"] == "status", "intent", ] = 3
df.loc[df["intent"] == "affirmation", "intent", ] = 4
df.loc[df["intent"] == "negation", "intent", ] = 5
from sklearn.feature_extraction.text import CountVectorizer
def get_top_n_bigram(corpus, n=None):
    vec = CountVectorizer(ngram_range=(2, 2)).fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]
common_words = get_top_n_bigram(df['query'], 20)
for word, freq in common_words:
    print(word, freq)
df3 = pd.DataFrame(common_words, columns = ['ReviewText' , 'count'])
df3.groupby('ReviewText').sum()['count'].sort_values(ascending=False).iplot(
    kind='bar', yTitle='Count', linecolor='black', title='Top 20 bigrams words indataset')
def get_top_n_trigram(corpus, n=None):
    vec = CountVectorizer(ngram_range=(3, 3)).fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]
common_words = get_top_n_trigram(df['query'], 20)
for word, freq in common_words:
    print(word, freq)
df5 = pd.DataFrame(common_words, columns = ['ReviewText' , 'count'])
df5.groupby('ReviewText').sum()['count'].sort_values(ascending=False).iplot(
    kind='bar', yTitle='Count', linecolor='black', title='Top 20 trigrams words in dataset')

from nltk.tokenize import sent_tokenize, word_tokenize 
import warnings 
  
warnings.filterwarnings(action = 'ignore') 
  
import gensim 
from gensim.models import Word2Vec 
my_file = open("text.txt","w+")
row,col=df.shape

for i in range(0,row):
    x=df['query'][i]
    my_file.write(x)
sample = open("/kaggle/working/text.txt", "r") 
s = sample.read() 
  
# Replaces escape character with space 
f = s.replace("\n", " ") 
data = [] 
  
# iterate through each sentence in the file 
for i in sent_tokenize(f): 
    temp = [] 
      
    # tokenize the sentence into words 
    for j in word_tokenize(i): 
        temp.append(j.lower()) 
  
    data.append(temp)
model1 = gensim.models.Word2Vec(data, min_count = 1,  
                              size = 100, window = 5) 
print("Cosine similarity between 'flight' " + 
               "and 'cancel' - CBOW : ", 
    model1.similarity('flight', 'cancel')) 
print("Cosine similarity between 'return' " + 
               "and 'cancel' - CBOW : ", 
    model1.similarity('return', 'cancel')) 
x=df['query']
y=df['intent']
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x.values,y.values,test_size=0.2,random_state=4)
from sklearn.linear_model import LogisticRegression
logisticRegr = LogisticRegression()
cv=CountVectorizer()
model=cv.fit_transform(x_train)
model.toarray()
model.shape
y_train.shape
from sklearn.feature_selection import SelectKBest, chi2

X_new = SelectKBest(chi2, k=100).fit_transform(model, y_train)
X_test=SelectKBest(chi2, k=100).fit_transform(X_test, y_test)
X_new.shape
logisticRegr.fit(X_new, y_train.astype('int'))
logisticRegr.score(X_new, y_train.astype('int'))
y_pred = logisticRegr.predict(X_new)

y_pred
from sklearn.metrics import confusion_matrix
confusion_matrix(y_train.astype('int'), y_pred)
from sklearn.metrics import precision_recall_fscore_support

precision_recall_fscore_support(y_train, y_pred, average='macro')