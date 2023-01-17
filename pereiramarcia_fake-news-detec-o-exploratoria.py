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
from nltk.stem.porter import PorterStemmer
from nltk.stem import PorterStemmer,WordNetLemmatizer
from nltk.stem import RSLPStemmer
from nltk import FreqDist
import gensim
from gensim.models import LdaModel
from gensim import models, corpora, similarities
import re
import os
import string
from string import punctuation
import time
from scipy.stats import entropy
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("darkgrid")
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud, STOPWORDS
#import tensorflow as tf
from zipfile import ZipFile
import pyLDAvis.gensim
from collections import defaultdict, Counter
from subprocess import check_output
from nltk import word_tokenize


# Carregando dataset de treino e de teste
train = pd.read_excel('../input/treino-noticias-portugues/train.xlsx')
test = pd.read_excel('../input/test-noticias-portugues/test.xlsx')
train_data = train.copy()
test_data = test.copy()
train_data.head()
# tranformando a coluna 'id' como índice
train_data = train_data.set_index('id', drop = True)
print(train_data.shape)
train_data.head()
# Verificando os valores missing
train_data.isnull().sum()
# Incluindo a palavra "Missing" nos dados 'na' na coluna Author e excluíndo linhas com dados 'na'
# de todo o dataset
train_data[['author']] = train_data[['author']].fillna(value = 'Missing')
train_data = train_data.dropna()
train_data.isnull().sum()
# Verificando a quantidade de observações restantes:
train_data.shape
# Subsets para testes
train_subset = train_data[0:20000]
train_subset.head()
train_subset.shape
# incluindo uma coluna com a 'length' do campo texto:
length = []
[length.append(len(str(text))) for text in train_subset['text']]
train_subset['length'] = length
train_subset.head()
#verificando o balanceamento da variável resposta:
train_subset['label'].value_counts().plot.bar()
#Tranformando os títulos, reais ou fakes, em listas separadas e verificando alguns padrões que podem diferenciá-los:

# verificando os títulos das mensagens reais:
tit_real=' '.join(list(train_subset[train_subset['label']==0]['title']))
print(tit_real)
# verificando os títulos das mensagens fake:
tit_fake=' '.join(list(train_subset[train_subset['label']==1]['title']))
print(tit_real)
# incluindo palavras no stopword

stop_words = stopwords.words('portuguese') #stop words in Português
newStopWords = ['The New York Times', 'de', 'do', 'para', 'no', 'na', 'nos' , 'nas', 'uma', 'um', 
                'umas','uns', 'a', 'Breitbart', 'New York','York', 'Times', 'The', 'New','the', 'The New',
               ' York times', ' Times', ' times', 'York Times', 'breitbart', 'times', 'Times', ' Breitbart',
               '- The New York Times', 'em', 'new', 'york', 'the', 'sobre', 'ser', 'sr', 'pode', 'disse', 'vai']
stop_words.extend(newStopWords)
print(stop_words)
# Separando os dados de treino em mensagens 'fake' e 'not fake', utilizando o label:
# 0 = verdadeiro   
# 1 = fake
msg_fake=train_subset[train_subset.label==1].copy()
msg_real=train_subset[train_subset.label==0].copy()
# Limpando e retirando as stopwords dos títulos das notícias verdadeiras

msg_real['title'] = msg_real['title'].str.lower()
msg_real['title'] = msg_real.title.apply(lambda x: x.translate(str.maketrans('', '', string.punctuation)))
new_real = msg_real['title'].str.split() # separa palavras do titulo
new_real=new_real.values.tolist()
corpus_title_real=[word for i in new_real for word in i] # separa as palavras todas
title_real_clean = [msg for msg in corpus_title_real if msg not in stop_words]
#[s for s in title_real_clean if "the" in s]

#Nuvem de palavras dos títulos das mensagens verdadeiras

real=' '.join(list(title_real_clean))
cloud_title_real_clean=WordCloud(width=512, height=512).generate(real)
plt.figure(figsize=(5,5),facecolor='k')
plt.imshow(cloud_title_real_clean)
plt.axis('off')
plt.tight_layout(pad=0)
plt.show()
# LImpando e retirando as stopwords dos títulos das notícias falsas
msg_fake['title'] = msg_fake['title'].str.lower()
msg_fake['title'] = msg_fake.title.apply(lambda x: x.translate(str.maketrans('', '', string.punctuation)))
new_fake = msg_fake['title'].str.split() # separa palavras do titulo
new_fake=new_fake.values.tolist()
corpus_title_fake=[word for i in new_fake for word in i] # separa as palavras todas
title_fake_clean = [msg for msg in corpus_title_fake if msg not in stop_words]
#[s for s in title_real_clean if "the" in s]
# Nuvem de palavras dos títulos das mensagens falsas

fake=' '.join(list(title_fake_clean))
cloud_title_fake_clean=WordCloud(width=512, height=512).generate(fake)
plt.figure(figsize=(5,5),facecolor='k')
plt.imshow(cloud_title_fake_clean)
plt.axis('off')
plt.tight_layout(pad=0)
plt.show()
# Análise do Corpus dos TEXTOS das mensagens VERDADEIRAS e Gráfico com as palavras mais frequentes:

msg_real['text'] = msg_real['text'].str.lower()
msg_real['text'] = msg_real.text.apply(lambda x: x.translate(str.maketrans('', '', string.punctuation)))
new = msg_real['text'].str.split()
new=new.values.tolist()
corpus_text_real=[word for i in new for word in i]

counter=Counter(corpus_text_real)
most=counter.most_common()
x, y= [], []
for word,count in most[:50]:
    if (word not in stop_words):
        x.append(word)
        y.append(count)
        
sns.barplot(x=y,y=x)
# Análise do Corpus dos TEXTOS das mensagens FAKE e gráfico com as palavras mais frequentes:

msg_fake['text'] = msg_fake.text.apply(lambda x: str(x))
msg_fake['text'] = msg_fake['text'].str.lower()
msg_fake['text'] = msg_fake.text.apply(lambda x: x.translate(str.maketrans('', '', string.punctuation)))
new = msg_fake['text'].str.split() 
new=new.values.tolist()
corpus_text_fake=[word for i in new for word in i]

counter=Counter(corpus_text_fake)
most=counter.most_common()
x, y= [], []
for word,count in most[:50]:
    if (word not in stop_words):
        x.append(word)
        y.append(count)
        
sns.barplot(x=y,y=x)
# Ngram são palavras que normalmente aparecem juntas, torna-se necessário a verificação das frequências destas
# expressões

# Função de análise de Ngram:

def get_top_ngram(corpus, n=None):
    vec = CountVectorizer(ngram_range=(n, n)).fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) 
                  for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:10]
# Ngram analysis corpus_title_real

msg_real['title'] = msg_real['title'].str.lower()
msg_real['title'] = msg_real.title.apply(lambda x: x.translate(str.maketrans('', '', string.punctuation)))
msg_real['title'] = msg_real['title'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop_words)]))

top_n_bigrams=get_top_ngram(msg_real['title'],2)[:10]
x,y=map(list,zip(*top_n_bigrams))
sns.barplot(x=y,y=x)
# Ngram analysis corpus_title_fake

msg_fake['title'] = msg_fake['title'].str.lower()
msg_fake['title'] = msg_fake.title.apply(lambda x: x.translate(str.maketrans('', '', string.punctuation)))
msg_fake['title'] = msg_fake['title'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop_words)]))

top_n_bigrams=get_top_ngram(msg_fake['title'],2)[:10]
x,y=map(list,zip(*top_n_bigrams))
sns.barplot(x=y,y=x)
# Ngram analysis corpus_text_fake

msg_fake['text'] = msg_fake['text'].str.lower()
msg_fake['text'] = msg_fake.text.apply(lambda x: x.translate(str.maketrans('', '', string.punctuation)))
msg_fake['text'] = msg_fake['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop_words)]))

top_n_bigrams=get_top_ngram(msg_fake['text'],2)[:10]
x,y=map(list,zip(*top_n_bigrams))
sns.barplot(x=y,y=x)
# Ngram analysis corpus_text_real

msg_real['text'] = msg_real['text'].str.lower()
msg_real['text'] = msg_real.text.apply(lambda x: x.translate(str.maketrans('', '', string.punctuation)))
msg_real['text'] = msg_real['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop_words)]))

top_n_bigrams=get_top_ngram(msg_real['text'],2)[:10]
x,y=map(list,zip(*top_n_bigrams))
sns.barplot(x=y,y=x)
#  Preprocess function for LDA

def preprocess_news(df,column):
    corpus=[]
    #stem=PorterStemmer()
    stem=nltk.stem.RSLPStemmer() # remove sufixos da lingua portuguesa
    lem=WordNetLemmatizer()
    for news in df[column]:
        words=[w for w in word_tokenize(news) if (w not in stop_words)]
        
        words=[lem.lemmatize(w) for w in words if len(w)>2]
        
        corpus.append(words)
    return corpus
#LDA for corpus_title_fake
corpus = preprocess_news(msg_fake,'title')
dic=gensim.corpora.Dictionary(corpus)
bow_corpus = [dic.doc2bow(doc) for doc in corpus]
lda_model = gensim.models.LdaMulticore(bow_corpus, 
                                   num_topics = 3, 
                                   id2word = dic,                                    
                                   passes = 10,
                                   workers = 2)
lda_model.show_topics()
#LDA for corpus_title_real
corpus = preprocess_news(msg_real,'title')
dic=gensim.corpora.Dictionary(corpus)
bow_corpus = [dic.doc2bow(doc) for doc in corpus]
lda_model = gensim.models.LdaMulticore(bow_corpus, 
                                   num_topics = 3, 
                                   id2word = dic,                                    
                                   passes = 10,
                                   workers = 2)
lda_model.show_topics()
#LDA for corpus_text_fake
corpus = preprocess_news(msg_fake,'text')
dic=gensim.corpora.Dictionary(corpus)
bow_corpus = [dic.doc2bow(doc) for doc in corpus]
lda_model = gensim.models.LdaMulticore(bow_corpus, 
                                   num_topics = 5, 
                                   id2word = dic,                                    
                                   passes = 10,
                                   workers = 2)
lda_model.show_topics()
lda_model.print_topics(0, 30)
lda_model.print_topics(1, 30)
lda_model.print_topics(2, 30)
lda_model.print_topics(3, 30)
lda_model.print_topics(4, 30)
lda_model.print_topics(5, 30)
#LDA for corpus_real
corpus = preprocess_news(msg_real,'text')
dic=gensim.corpora.Dictionary(corpus)
bow_corpus = [dic.doc2bow(doc) for doc in corpus]
lda_model = gensim.models.LdaMulticore(bow_corpus, 
                                   num_topics = 5, 
                                   id2word = dic,                                    
                                   passes = 10,
                                   workers = 2)
lda_model.show_topics()
lda_model.print_topics(0, 30)
lda_model.print_topics(1, 30)
lda_model.print_topics(2, 30)
lda_model.print_topics(3, 30)
lda_model.print_topics(4, 30)
lda_model.print_topics(5, 30)
