# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# Importando as bibliotecas necessárias:
%matplotlib inline
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
import gensim
from gensim.models import LdaModel
from gensim import models, corpora, similarities
import re
from nltk.stem.porter import PorterStemmer
from nltk.stem import RSLPStemmer
import time
from nltk import FreqDist
from scipy.stats import entropy
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("darkgrid")
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
import os # accessing directory structure
from wordcloud import WordCloud, STOPWORDS
import pandas as pd
import tensorflow as tf
import os
import re
import numpy as np
import string
from string import punctuation
from zipfile import ZipFile
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
# Importando as bibliotecas necessárias:
%matplotlib inline
import nltk
from nltk.corpus import stopwords
import gensim
from gensim.models import LdaModel
from gensim import models, corpora, similarities
import re
from nltk.stem import PorterStemmer,WordNetLemmatizer
from nltk.tokenize import word_tokenize
import time
from nltk import FreqDist
from scipy.stats import entropy
import seaborn as sns
sns.set_style("darkgrid")
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
import os # accessing directory structure
from wordcloud import WordCloud, STOPWORDS
import pyLDAvis.gensim
from collections import defaultdict, Counter
from sklearn.feature_extraction.text import CountVectorizer
# importing neural network libraries
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Embedding, GRU, LSTM, RNN, SpatialDropout1D
train = pd.read_csv('../input/fake-news/train.csv')
test = pd.read_csv('../input/fake-news/test.csv')
train_data = train.copy()
test_data = test.copy()
train_data = train_data.set_index('id', drop = True)
print(train_data.shape)
train_data.head()
# checking for missing values
train_data.isnull().sum()
# dropping missing values from text columns alone. 
train_data[['title', 'author']] = train_data[['title', 'author']].fillna(value = 'Missing')
train_data = train_data.dropna()
train_data.isnull().sum()
# incluindo uma coluna com a 'length' do campo texto:
length = []
[length.append(len(str(text))) for text in train_data['text']]
train_data['length'] = length
train_data.head()
#verificando o balanceamento da variável resposta:
train_data['label'].value_counts().plot.bar()
# nuvem de palavras dos títulos dos artigos confiáveis
real=' '.join(list(train_data[train_data['label']==0]['title']))
real=WordCloud(width=512, height=512).generate(real)
plt.figure(figsize=(5,5),facecolor='k')
plt.imshow(real)
plt.axis('off')
plt.tight_layout(pad=0)
plt.show()
# nuvem de palavras dos títulos dos artigos "fake"
real=' '.join(list(train_data[train_data['label']==1]['title']))
real=WordCloud(width=512, height=512).generate(real)
plt.figure(figsize=(5,5),facecolor='k')
plt.imshow(real)
plt.axis('off')
plt.tight_layout(pad=0)
plt.show()
# Remove stopwords from title and text by label
msg_fake=train_data[train_data.label==1].copy()
msg_not_fake=train_data[train_data.label==0].copy()
msg_fake.head()
# Corpus analysis_TÍTULO FAKE
stop=set(stopwords.words('english'))
msg_fake['title'] = msg_fake['title'].str.lower()
msg_fake['title'] = msg_fake.title.apply(lambda x: x.translate(str.maketrans('', '', string.punctuation)))
new = msg_fake['title'].str.split()
new=new.values.tolist()
corpus_title_fake=[word for i in new for word in i]

counter=Counter(corpus_title_fake)
most=counter.most_common()
x, y= [], []
for word,count in most[:25]:
    if (word not in stop):
        x.append(word)
        y.append(count)
        
sns.barplot(x=y,y=x)
# Corpus analysis corpus_TEXTO_FAKE
stop=set(stopwords.words('english'))
msg_fake['text'] = msg_fake['text'].str.lower()
msg_fake['text'] = msg_fake.text.apply(lambda x: x.translate(str.maketrans('', '', string.punctuation)))
new = msg_fake['text'].str.split()
new=new.values.tolist()
corpus_text_fake=[word for i in new for word in i]

counter=Counter(corpus_text_fake)
most=counter.most_common()
x, y= [], []
for word,count in most[:50]:
    if (word not in stop):
        x.append(word)
        y.append(count)
        
sns.barplot(x=y,y=x)
# Corpus analysis TÍTULO NÃO FAKE
stop=set(stopwords.words('english'))
msg_not_fake['title'] = msg_not_fake['title'].str.lower()
msg_not_fake['title'] = msg_not_fake.title.apply(lambda x: x.translate(str.maketrans('', '', string.punctuation)))
new = msg_not_fake['title'].str.split()
new=new.values.tolist()
corpus_title_not_fake=[word for i in new for word in i]

counter=Counter(corpus_title_not_fake)
most=counter.most_common()
x, y= [], []
for word,count in most[:50]:
    if (word not in stop):
        x.append(word)
        y.append(count)
        
sns.barplot(x=y,y=x)
# Corpus analysis TEXTO NÃO FAKE
stop=set(stopwords.words('english'))
msg_not_fake['text'] = msg_not_fake['text'].str.lower()
msg_not_fake['text'] = msg_not_fake.text.apply(lambda x: x.translate(str.maketrans('', '', string.punctuation)))
new = msg_not_fake['text'].str.split()
new=new.values.tolist()
corpus_text_not_fake=[word for i in new for word in i]

counter=Counter(corpus_text_not_fake)
most=counter.most_common()
x, y= [], []
for word,count in most[:50]:
    if (word not in stop):
        x.append(word)
        y.append(count)
        
sns.barplot(x=y,y=x)
# Ngram analysis function

def get_top_ngram(corpus, n=None):
    vec = CountVectorizer(ngram_range=(n, n)).fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) 
                  for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:10]
# Ngram analysis corpus_title_fake

msg_fake['title'] = msg_fake['title'].str.lower()
msg_fake['title'] = msg_fake.title.apply(lambda x: x.translate(str.maketrans('', '', string.punctuation)))
msg_fake['title'] = msg_fake['title'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))

top_n_bigrams=get_top_ngram(msg_fake['title'],2)[:10]
x,y=map(list,zip(*top_n_bigrams))
sns.barplot(x=y,y=x)
# Ngram analysis corpus_text_fake

msg_fake['text'] = msg_fake['text'].str.lower()
msg_fake['text'] = msg_fake.text.apply(lambda x: x.translate(str.maketrans('', '', string.punctuation)))
msg_fake['text'] = msg_fake['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))

top_n_bigrams=get_top_ngram(msg_fake['text'],2)[:10]
x,y=map(list,zip(*top_n_bigrams))
sns.barplot(x=y,y=x)
# Ngram analysis corpus_title_not_fake

msg_not_fake['title'] = msg_not_fake['title'].str.lower()
msg_not_fake['title'] = msg_not_fake.title.apply(lambda x: x.translate(str.maketrans('', '', string.punctuation)))
msg_not_fake['title'] = msg_not_fake['title'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))

top_n_bigrams=get_top_ngram(msg_not_fake['title'],2)[:10]
x,y=map(list,zip(*top_n_bigrams))
sns.barplot(x=y,y=x)
# Ngram analysis corpus_text_not_fake

msg_not_fake['text'] = msg_not_fake['text'].str.lower()
msg_not_fake['text'] = msg_not_fake.text.apply(lambda x: x.translate(str.maketrans('', '', string.punctuation)))
msg_not_fake['text'] = msg_not_fake['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))

top_n_bigrams=get_top_ngram(msg_not_fake['text'],2)[:10]
x,y=map(list,zip(*top_n_bigrams))
sns.barplot(x=y,y=x)
#  Preprocess function for LDA
nltk.download('punkt')
nltk.download('wordnet')

def preprocess_news(df,column):
    corpus=[]
    stem=PorterStemmer()
    lem=WordNetLemmatizer()
    for news in df[column]:
        words=[w for w in word_tokenize(news) if (w not in stop)]
        
        words=[lem.lemmatize(w) for w in words if len(w)>2]
        
        corpus.append(words)
    return corpus
#LDA for corpus_title_fake
corpus = preprocess_news(msg_fake,'title')
dic=gensim.corpora.Dictionary(corpus)
bow_corpus = [dic.doc2bow(doc) for doc in corpus]
lda_model = gensim.models.LdaMulticore(bow_corpus, 
                                   num_topics = 4, 
                                   id2word = dic,                                    
                                   passes = 10,
                                   workers = 2)
lda_model.show_topics()
#Vizual for corpus_title_fake
pyLDAvis.enable_notebook()
vis = pyLDAvis.gensim.prepare(lda_model, bow_corpus, dic)
pyLDAvis.save_html(vis, 'lda_fake_title.html')
#LDA for corpus_title_not_fake
corpus = preprocess_news(msg_not_fake,'title')
dic=gensim.corpora.Dictionary(corpus)
bow_corpus = [dic.doc2bow(doc) for doc in corpus]
lda_model = gensim.models.LdaMulticore(bow_corpus, 
                                   num_topics = 4, 
                                   id2word = dic,                                    
                                   passes = 10,
                                   workers = 2)
#LDA for corpus_text_fake
corpus = preprocess_news(msg_fake,'text')
dic=gensim.corpora.Dictionary(corpus)
bow_corpus = [dic.doc2bow(doc) for doc in corpus]
lda_model = gensim.models.LdaMulticore(bow_corpus, 
                                   num_topics = 4, 
                                   id2word = dic,                                    
                                   passes = 10,
                                   workers = 2)
#LDA for corpus_text_not_fake
corpus = preprocess_news(msg_not_fake,'text')
dic=gensim.corpora.Dictionary(corpus)
bow_corpus = [dic.doc2bow(doc) for doc in corpus]
lda_model = gensim.models.LdaMulticore(bow_corpus, 
                                   num_topics = 4, 
                                   id2word = dic,                                    
                                   passes = 10,
                                   workers = 2)
