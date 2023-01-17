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
import pandas as pd
import numpy as np
df =pd.read_csv('/kaggle/input/question-pairs-dataset/questions.csv',nrows=5000)
df.head()
df=df.dropna()
import matplotlib.pyplot as plt
df.is_duplicate.value_counts()
df.is_duplicate.value_counts().plot(kind='bar')
plt.show()
df.info()
q = df[['question1','question2']]
q.head()
import nltk
from nltk.tokenize import word_tokenize
q['tokenized1'] = q['question1'].apply(word_tokenize)
q.head()
q['tokenized2'] = q['question2'].apply(word_tokenize)
q.head()
q.sample(frac=0.05)
q['lower1'] = q['tokenized1'].apply(lambda x: [word.lower() for word in x])
q.head()
q['lower2'] = q['tokenized2'].apply(lambda x: [word.lower() for word in x])
q.head()
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
q['lower1'] = q['lower1'].apply(lambda x: [word for word in x if word not in stop_words])
q.head()

q['lower2'] = q['lower2'].apply(lambda x: [word for word in x if word not in stop_words])
q.head()

df.head()
q['pos_tags1'] = q['lower1'].apply(nltk.tag.pos_tag)
q.head()
q['pos_tags2'] = q['lower2'].apply(nltk.tag.pos_tag)
q.head()

from nltk.corpus import wordnet
def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN
q['wordnet_pos1'] = q['pos_tags1'].apply(lambda x: [(word, get_wordnet_pos(pos_tag1)) for (word, pos_tag1) in x])
q.head()
q['wordnet_pos2'] = q['pos_tags2'].apply(lambda x: [(word, get_wordnet_pos(pos_tag2)) for (word, pos_tag2) in x])
q.head()
from nltk.stem import WordNetLemmatizer
wnl = WordNetLemmatizer()
q['lemmatized1'] = q['wordnet_pos1'].apply(lambda x: [wnl.lemmatize(word, tag) for word, tag in x])
q.head()

q['lemmatized2'] = q['wordnet_pos2'].apply(lambda x: [wnl.lemmatize(word, tag) for word, tag in x])
q.head()

import seaborn as sns
from collections import  Counter

def plot_top_non_stopwords_barchart(text):
    stop=set(stopwords.words('english'))
    
    new= text.str.split()
    new=new.values.tolist()
    corpus=[word for i in new for word in i]

    counter=Counter(corpus)
    most=counter.most_common()
    x, y=[], []
    for word,count in most[:50]:
        if (word not in stop):
            x.append(word)
            y.append(count)
    plt.figure(figsize=(10,10))
    sns.barplot(x=y,y=x)

plot_top_non_stopwords_barchart(q['question1'])


import seaborn as sns
from collections import  Counter

def plot_top_non_stopwords_barchart(text):
    stop=set(stopwords.words('english'))
    
    new= text.str.split()
    new=new.values.tolist()
    corpus=[word for i in new for word in i]

    counter=Counter(corpus)
    most=counter.most_common()
    x, y=[], []
    for word,count in most[:50]:
        if (word not in stop):
            x.append(word)
            y.append(count)
    plt.figure(figsize=(10,10))
    sns.barplot(x=y,y=x)

plot_top_non_stopwords_barchart(q['question2'])


import spacy
nlp = spacy.load('en_core_web_sm')
nlp
from spacy import displacy

for sentence in q['question1'].sample(5, random_state = 5):
  sentence_doc = nlp(sentence)
  
  displacy.render(sentence_doc, style='dep', jupyter=True)
  print("Sentence is: ", sentence_doc)
from spacy import displacy

for sentence in q['question2'].sample(5, random_state = 5):
  sentence_doc = nlp(sentence)
  
  displacy.render(sentence_doc, style='dep', jupyter=True)
  print("Sentence is: ", sentence_doc)
for sentence in q['question1'].sample(5, random_state = 5):
  print("Sentence is: ", sentence)
  sentence_doc = nlp(sentence)

  for chunk in sentence_doc.noun_chunks:
    print ("Chunked noun phrases found: ",chunk)
  print()
for sentence in q['question2'].sample(5, random_state = 5):
  print("Sentence is: ", sentence)
  sentence_doc = nlp(sentence)

  for chunk in sentence_doc.noun_chunks:
    print ("Chunked noun phrases found: ",chunk)
  print()
for sentence in q['question1'].sample(5, random_state = 5):
  print("Sentence is: ", sentence)
  sentence_doc = nlp(sentence)
  displacy.render(sentence_doc,style='ent',jupyter=True)
  print()
for sentence in q['question2'].sample(5, random_state = 5):
  print("Sentence is: ", sentence)
  sentence_doc = nlp(sentence)
  displacy.render(sentence_doc,style='ent',jupyter=True)
  print()
from tqdm import tqdm, tqdm_notebook

nlp = spacy.load('en',
                 disable=['parser', 
                          'tagger',
                          'textcat'])
df.head()
q.head()

from nltk.util import ngrams
from sklearn.feature_extraction.text import CountVectorizer

def plot_top_ngrams_barchart(text, n=2):
    stop=set(stopwords.words('english'))

    new= text.str.split()
    new=new.values.tolist()
    corpus=[word for i in new for word in i]

    def _get_top_ngram(corpus, n=None):
        vec = CountVectorizer(ngram_range=(n, n)).fit(corpus)
        bag_of_words = vec.transform(corpus)
        sum_words = bag_of_words.sum(axis=0) 
        words_freq = [(word, sum_words[0, idx]) 
                      for word, idx in vec.vocabulary_.items()]
        words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
        return words_freq[:20]

    top_n_bigrams=_get_top_ngram(text,n)[:20]
    x,y=map(list,zip(*top_n_bigrams))
    plt.figure(figsize=(10,10))
    plt.xlabel("Bi-gram Frequency")
    plt.ylabel("Top 20 bi-grams mentioned in News Title")
    sns.barplot(x=y,y=x)


plot_top_ngrams_barchart(df['question1'],2)
def plot_top_ngrams_barchart(text, n=2):
    stop=set(stopwords.words('english'))

    new= text.str.split()
    new=new.values.tolist()
    corpus=[word for i in new for word in i]

    def _get_top_ngram(corpus, n=None):
        vec = CountVectorizer(ngram_range=(n, n)).fit(corpus)
        bag_of_words = vec.transform(corpus)
        sum_words = bag_of_words.sum(axis=0) 
        words_freq = [(word, sum_words[0, idx]) 
                      for word, idx in vec.vocabulary_.items()]
        words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
        return words_freq[:20]

    top_n_bigrams=_get_top_ngram(text,n)[:20]
    x,y=map(list,zip(*top_n_bigrams))
    plt.figure(figsize=(10,10))
    plt.xlabel("Bi-gram Frequency")
    plt.ylabel("Top 20 bi-grams mentioned in News Title")
    sns.barplot(x=y,y=x)


plot_top_ngrams_barchart(df['question2'],2)
q.head()
q['p'] = q['lower1'].apply(lambda x: [word for word in x if word not in stop_words])
q['p'] = [' '.join(map(str, l)) for l in q['p']]

q.sample(10, random_state = 5)
q['q'] = q['lower2'].apply(lambda x: [word for word in x if word not in stop_words])
q['q'] = [' '.join(map(str, l)) for l in q['q']]

q.sample(10, random_state = 5)

X = q[['p','q']]
X.head()
X.info()
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
search_terms = 'fruit and vegetables'
documents = ['cars drive on the road', 'tomatoes are actually fruit']

doc_vectors = TfidfVectorizer().fit_transform([search_terms] + documents)

cosine_similarities = linear_kernel(doc_vectors[0:1], doc_vectors).flatten()
document_scores = [item.item() for item in cosine_similarities[1:]]
print(document_scores)
text_content1 = X['p']
vector = TfidfVectorizer(max_df=0.3,         # drop words that occur in more than X percent of documents
                             #min_df=8,      # only use words that appear at least X times
                             stop_words='english', # remove stop words
                             lowercase=True, # Convert everything to lower case 
                             use_idf=True,   # Use idf
                             norm=u'l2',     # Normalization
                             smooth_idf=True # Prevents divide-by-zero errors
                            )
content_subset1 = text_content1[0:100000]
tfidf_subset1 = vector.fit_transform(content_subset1)
tfidf_subset1 = tfidf_subset1.toarray()

vocab = vector.get_feature_names()
X1= pd.DataFrame(np.round(tfidf_subset1, 2), columns=vocab)
X1.head()
text_content2 = X['q']
vector = TfidfVectorizer(max_df=0.3,         # drop words that occur in more than X percent of documents
                             #min_df=8,      # only use words that appear at least X times
                             stop_words='english', # remove stop words
                             lowercase=True, # Convert everything to lower case 
                             use_idf=True,   # Use idf
                             norm=u'l2',     # Normalization
                             smooth_idf=True # Prevents divide-by-zero errors
                            )
content_subset2 = text_content2[0:100000]
tfidf_subset2 = vector.fit_transform(content_subset2)
tfidf_subset2 = tfidf_subset2.toarray()

vocab = vector.get_feature_names()
X2= pd.DataFrame(np.round(tfidf_subset2, 2), columns=vocab)
X2.head()
X.head()
from nltk.corpus import stopwords

stops = set(stopwords.words("english"))

def word_match_share(row):
    q1words = {}
    q2words = {}
    for word in str(row['question1']).lower().split():
        if word not in stops:
            q1words[word] = 1
    for word in str(row['question2']).lower().split():
        if word not in stops:
            q2words[word] = 1
    if len(q1words) == 0 or len(q2words) == 0:
        # The computer-generated chaff includes a few questions that are nothing but stopwords
        return 0
    shared_words_in_q1 = [w for w in q1words.keys() if w in q2words]
    shared_words_in_q2 = [w for w in q2words.keys() if w in q1words]
    R = (len(shared_words_in_q1) + len(shared_words_in_q2))/(len(q1words) + len(q2words))
    return R


train_word_match = df.apply(word_match_share, axis=1, raw=True)
train_word_match
df_q = pd.Series(df['question1'].tolist() + df['question2'].tolist()).astype(str)
df_q
from collections import Counter

# If a word appears only once, we ignore it completely (likely a typo)
# Epsilon defines a smoothing constant, which makes the effect of extremely rare words smaller
def get_weight(count, eps=10000, min_count=2):
    if count < min_count:
        return 0
    else:
        return 1 / (count + eps)

eps = 5000 
words = (" ".join(df_q)).lower().split()
counts = Counter(words)
weights = {word: get_weight(count) for word, count in counts.items()}
print('Most common words and weights: \n')
print(sorted(weights.items(), key=lambda x: x[1] if x[1] > 0 else 9999)[:10])
print('\nLeast common words and weights: ')
(sorted(weights.items(), key=lambda x: x[1], reverse=True)[:10])
def tfidf_word_match_share(row):
    q1words = {}
    q2words = {}
    for word in str(row['question1']).lower().split():
        if word not in stops:
            q1words[word] = 1
    for word in str(row['question2']).lower().split():
        if word not in stops:
            q2words[word] = 1
    if len(q1words) == 0 or len(q2words) == 0:
        # The computer-generated chaff includes a few questions that are nothing but stopwords
        return 0
    
    shared_weights = [weights.get(w, 0) for w in q1words.keys() if w in q2words] + [weights.get(w, 0) for w in q2words.keys() if w in q1words]
    total_weights = [weights.get(w, 0) for w in q1words] + [weights.get(w, 0) for w in q2words]
    
    R = np.sum(shared_weights) / np.sum(total_weights)
    return R
tf_idf_word_match_share = df.apply(tfidf_word_match_share, axis=1, raw=True)
tf_idf_word_match_share
plt.figure(figsize=(15, 5))
plt.hist(train_word_match[df['is_duplicate'] == 0], bins=20, label='Not Duplicate')
plt.hist(train_word_match[df['is_duplicate'] == 1], bins=20,  alpha=0.7, label='Duplicate')
plt.legend()
plt.title('Label distribution over word_match_share', fontsize=15)
plt.xlabel('word_match_share', fontsize=15)

plt.figure(figsize=(15, 5))
plt.hist(tf_idf_word_match_share[df['is_duplicate'] == 0], bins=20, label='Not Duplicate')
plt.hist(tf_idf_word_match_share[df['is_duplicate'] == 1], bins=20,  alpha=0.7, label='Duplicate')
plt.legend()
plt.title('Label distribution over tfidf word_match_share', fontsize=15)
plt.xlabel('word_match_share', fontsize=15)

from sklearn.metrics import roc_auc_score
print('Original AUC:', roc_auc_score(df['is_duplicate'], train_word_match))
print('   TFIDF AUC:', roc_auc_score(df['is_duplicate'], tf_idf_word_match_share.fillna(0)))

X = tf_idf_word_match_share

np.where(X.values >= np.finfo(np.float64).max)
X=X.to_numpy()

type(X)
np.shape(X)
X=X.reshape(-1,1)
y= df['is_duplicate'].values
type(y)
y=y.reshape(-1,1)

from sklearn.model_selection import train_test_split

x_train, x_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)
import xgboost as xgb

# Set our parameters for xgboost
params = {}
params['objective'] = 'binary:logistic'
params['eval_metric'] = 'logloss'
params['eta'] = 0.02
params['max_depth'] = 44

d_train = xgb.DMatrix(x_train, label=y_train)
d_valid = xgb.DMatrix(x_valid, label=y_valid)

watchlist = [(d_train, 'train'), (d_valid, 'valid')]

bst = xgb.train(params, d_train, 400, watchlist, early_stopping_rounds=50, verbose_eval=10)
