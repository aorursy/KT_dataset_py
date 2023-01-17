# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/all.csv')
df.head(10)
df.shape
content = df['content'].tolist()[:2]
content
df.groupby('type').count()
import matplotlib as mpl
import matplotlib.pyplot as plt
%matplotlib inline

from subprocess import check_output
from wordcloud import WordCloud, STOPWORDS

stopwords = set(STOPWORDS)
wordcloud = WordCloud(
                          background_color='black',
                          stopwords=stopwords,
                          max_words=200,
                          max_font_size=40, 
                          random_state=42
                         ).generate(str(df[df['type']=='Mythology & Folklore']['poem name']))

fig = plt.figure(1,figsize=(12,18))
plt.imshow(wordcloud)
plt.axis('off')
plt.show()
stopwords = set(STOPWORDS)
wordcloud = WordCloud(
                          background_color='white',
                          stopwords=stopwords,
                          max_words=200,
                          max_font_size=40, 
                          random_state=42
                         ).generate(str(df[df['type']=='Love']['poem name']))

fig = plt.figure(1,figsize=(12,18))
plt.imshow(wordcloud)
plt.axis('off')
plt.show()
stopwords = set(STOPWORDS)
wordcloud = WordCloud(
                          background_color='white',
                          stopwords=stopwords,
                          max_words=200,
                          max_font_size=40, 
                          random_state=42
                         ).generate(str(df[df['type']=='Nature']['poem name']))

fig = plt.figure(1,figsize=(12,18))
plt.imshow(wordcloud)
plt.axis('off')
plt.show()
author_count = df['author'].value_counts().head(10)
author_count
name =['WILLIAM SHAKESPEARE' ,'SIR PHILIP SIDNEY','JOHN DONNE' ,'EDMUND SPENSER' ,'WILLIAM BUTLER YEATS',' SIR THOMAS WYATT', 'CARL SANDBURG',' EZRA POUND', 'THOMAS CAMPION','HART CRANE']
count = [71,42,41,34,26,22,16,16,15,14]
import matplotlib.pyplot as plt
index = np.arange(len(name))
plt.bar(index, count)
plt.xlabel('author', fontsize=5)
plt.ylabel('No of poems', fontsize=5)
plt.xticks(index, name, fontsize=5, rotation=30)
plt.title('author wise count of poems')
plt.show()
import re,string
import nltk 
from nltk import word_tokenize
from nltk.corpus import stopwords
stops = set(stopwords.words("english"))
def preprocess(data):
    data = data.lower()
    data = re.sub(r'[^\x00-\x7f]',r' ',data)
    data = data.replace('\r','')
    data = data.replace('\n','')
    data = data.replace('  ','')
    data = data.replace('\'','')
    data = re.sub("["+string.punctuation+"]", " ", data)
    words = [word for word in data.split() if word not in stops]
    return " ".join(words)

from nltk.tokenize import sent_tokenize, word_tokenize
data_list = pd.Series([word_tokenize(preprocess(data)) for data in df['content']])
data_list.head(10)
from gensim.models import word2vec
num_features = 300    # Word vector dimensionality                      
min_word_count = 40   # Minimum word count                        
num_workers = 4       # Number of threads to run in parallel
context = 10          # Context window size                                                                                    
downsampling = 1e-3   # Downsample setting for frequent words
model = word2vec.Word2Vec(data_list, workers=num_workers,size=num_features, 
                          min_count = min_word_count,
                          window = context, sample = downsampling)


