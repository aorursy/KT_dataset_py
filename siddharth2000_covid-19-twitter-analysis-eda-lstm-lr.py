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
# Importing the libraries

import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.offline as pyoff
import plotly.graph_objs as go
import nltk
from collections import Counter

from plotly import graph_objs as go
from sklearn import preprocessing 
from sklearn.preprocessing import LabelBinarizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from wordcloud import WordCloud,STOPWORDS
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize,sent_tokenize
from bs4 import BeautifulSoup
import re,string,unicodedata
from keras.preprocessing import text, sequence
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
from sklearn.model_selection import train_test_split
from string import punctuation
from nltk import pos_tag
from nltk.corpus import wordnet
from bs4 import BeautifulSoup
from tqdm import tqdm
import re
import nltk
import gensim

from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import classification_report

import keras
from keras.models import Sequential
from keras.layers import Dense,Embedding,LSTM,Dropout, Bidirectional, Conv2D
from keras.callbacks import ReduceLROnPlateau
import tensorflow as tf
import transformers
from tokenizers import BertWordPieceTokenizer
from keras.layers import LSTM,Dense,Bidirectional,Input
from keras.models import Model
import torch
import transformers
df = pd.read_csv('/kaggle/input/covid-19-nlp-text-classification/Corona_NLP_train.csv', encoding='latin-1')
#test = pd.read_csv('/kaggle/input/covid-19-nlp-text-classification/Corona_NLP_test.csv')

df.head()
sns.countplot(df['Sentiment'])
df.info()
df['Location'].isna().sum()
location = df['Location'].value_counts().nlargest(n=15)

fig = px.bar(y=location.values,
       x=location.index,
       orientation='v',
       color=location.index,
       text=location.values,
       color_discrete_sequence= px.colors.qualitative.Bold)

fig.update_traces(texttemplate='%{text:.2s}', 
                  textposition='outside', 
                  marker_line_color='rgb(8,48,107)', 
                  marker_line_width=1.5, 
                  opacity=0.7)

fig.update_layout(width=1000, 
                  showlegend=False, 
                  xaxis_title="Location",
                  yaxis_title="Count",
                  title="Top 15 Locations with tweet count")
fig.show()
# Get all hashtags

def extract_hash_tags(s):
    hashes = re.findall(r"#(\w+)", s)
    return " ".join(hashes)
df['hashtags'] = df['OriginalTweet'].apply(lambda x : extract_hash_tags(x))
allHashTags = list(df[(df['hashtags'] != None) & (df['hashtags'] != "")]['hashtags'])
allHashTags = [tag.lower() for tag in allHashTags]
hash_df = dict(Counter(allHashTags))
top_hash_df = pd.DataFrame(list(hash_df.items()),columns = ['word','count']).reset_index(drop=True).sort_values('count',ascending=False)[:20]
top_hash_df.head()
fig = px.bar(x=top_hash_df['word'],y=top_hash_df['count'],
       orientation='v',
       color=top_hash_df['word'],
       text=top_hash_df['count'],
       color_discrete_sequence= px.colors.qualitative.Bold)

fig.update_traces(texttemplate='%{text:.2s}', 
                  textposition='outside', 
                  marker_line_color='rgb(8,48,107)', 
                  marker_line_width=1.5, 
                  opacity=0.7)

fig.update_layout(width=1000, 
                  showlegend=False, 
                  xaxis_title="Word",
                  yaxis_title="Count",
                  title="Top #hashtags in Covid19 Tweets")
fig.show()
# Get all mentions

def get_mentions(s):
    mentions = re.findall("(?<![@\w])@(\w{1,25})", s)
    return " ".join(mentions)
df['mentions'] = df['OriginalTweet'].apply(lambda x : get_mentions(x))
df['OriginalTweet'][0]
df['mentions'][0]
allMentions = list(df[(df['mentions'] != None) & (df['mentions'] != "")]['mentions'])
allMentions = [tag.lower() for tag in allMentions]
mentions_df = dict(Counter(allMentions))
top_mentions_df = pd.DataFrame(list(mentions_df.items()),columns = ['word','count']).reset_index(drop=True).sort_values('count',ascending=False)[:20]
top_mentions_df.head()
fig = px.bar(x=top_mentions_df['word'],y=top_mentions_df['count'],
       orientation='v',
       color=top_mentions_df['word'],
       text=top_mentions_df['count'],
       color_discrete_sequence= px.colors.qualitative.Bold)

fig.update_traces(texttemplate='%{text:.2s}', 
                  textposition='outside', 
                  marker_line_color='rgb(8,48,107)', 
                  marker_line_width=1.5, 
                  opacity=0.7)

fig.update_layout(width=1000, 
                  showlegend=False, 
                  xaxis_title="Word",
                  yaxis_title="Count",
                  title="Top #hashtags in Covid19 Tweets")
fig.show()
# Data Cleaning
stop = set(stopwords.words('english'))

def cleaner(phrase):
    phrase = re.sub(r"won't", "will not", phrase)
    phrase = re.sub(r"can't", 'can not', phrase)
  
  # general
    phrase = re.sub(r"n\'t"," not", phrase)
    phrase = re.sub(r"\'re'"," are", phrase)
    phrase = re.sub(r"\'s"," is", phrase)
    phrase = re.sub(r"\'ll"," will", phrase)
    phrase = re.sub(r"\'d"," would", phrase)
    phrase = re.sub(r"\'t"," not", phrase)
    phrase = re.sub(r"\'ve"," have", phrase)
    phrase = re.sub(r"\'m"," am", phrase)
    
    return phrase

cleaned_title = []

for sentance in tqdm(df['OriginalTweet'].values):
    sentance = str(sentance)
    sentance = re.sub(r"http\S+", "", sentance)
    sentance = BeautifulSoup(sentance, 'lxml').get_text()
    sentance = cleaner(sentance)
    sentance = re.sub(r'[?|!|\'|"|#|+]', r'', sentance)
    sentance = re.sub('<.*?>','',sentance)
    sentance = re.sub(r'@\w+','',sentance)
    sentance = re.sub(r'#\w+','',sentance)
    sentance = re.sub(r'[0-9]+','',sentance)
    sentance = re.sub(r'[0-9]+','',sentance)
    sentance = re.sub("\S*\d\S*", "", sentance).strip()
    sentance = re.sub('[^A-Za-z]+', ' ', sentance)
    sentance = ' '.join(e.lower() for e in sentance.split() if e.lower() not in stop)
    cleaned_title.append(sentance.strip())
    
df['text'] = cleaned_title
df.head()
# WordClouds
# Text that is displaying a positive sentiment
plt.figure(figsize = (20,20)) 
wc = WordCloud(max_words = 2000 , width = 1600 , height = 800).generate(" ".join(df[df.Sentiment == 'Positive'].text))
plt.imshow(wc , interpolation = 'bilinear')
# Text that is displaying a negative sentiment

plt.figure(figsize = (20,20)) 
wc = WordCloud(max_words = 2000 , width = 1600 , height = 800).generate(" ".join(df[df.Sentiment == 'Negative'].text))
plt.imshow(wc , interpolation = 'bilinear')
# Text that is displaying a neutral sentiment

plt.figure(figsize = (20,20)) 
wc = WordCloud(max_words = 2000 , width = 1600 , height = 800).generate(" ".join(df[df.Sentiment == 'Neutral'].text))
plt.imshow(wc , interpolation = 'bilinear')
# Continuing with some n-gram analysis

def basic_clean(text):
  """
  A simple function to clean up the data. All the words that
  are not designated as a stop word is then lemmatized after
  encoding and basic regex parsing are performed.
  """
  wnl = nltk.stem.WordNetLemmatizer()
  stopwords = nltk.corpus.stopwords.words('english')
  text = (unicodedata.normalize('NFKD', text)
    .encode('ascii', 'ignore')
    .decode('utf-8', 'ignore')
    .lower())
  words = re.sub(r'[^\w\s]', '', text).split()
  return [wnl.lemmatize(word) for word in words if word not in stopwords]
# Uni-grams for Tweets

HQ_words = basic_clean(''.join(str(df['text'].tolist())))
unigram_HQ = (pd.Series(nltk.ngrams(HQ_words, 1)).value_counts())[:20]
unigram_HQ = pd.DataFrame(unigram_HQ)
unigram_HQ['idx'] = unigram_HQ.index
unigram_HQ['idx'] = unigram_HQ.apply(lambda x: '('+x['idx'][0]+')',axis=1)
plot_data = [
    go.Bar(
        x=unigram_HQ['idx'],
        y=unigram_HQ[0],
        marker = dict(
            color = 'Blue'
        )
    )
]
plot_layout = go.Layout(
        title='Top 20 uni-grams from Covid-19 Tweets',
        yaxis_title='Count',
        xaxis_title='Uni-gram',
        plot_bgcolor='rgba(0,0,0,0)'
    )
fig = go.Figure(data=plot_data, layout=plot_layout)
pyoff.iplot(fig)

# Bi-grams for Tweets

HQ_words = basic_clean(''.join(str(df[df['Sentiment']=='Negative']['text'].tolist())))
bigram_HQ = (pd.Series(nltk.ngrams(HQ_words, 2)).value_counts())[:20]
bigram_HQ = pd.DataFrame(bigram_HQ)
bigram_HQ['idx'] = bigram_HQ.index
bigram_HQ['idx'] = bigram_HQ.apply(lambda x: '('+x['idx'][0]+', '+x['idx'][1]+')',axis=1)
plot_data = [
    go.Bar(
        x=bigram_HQ['idx'],
        y=bigram_HQ[0],
        marker = dict(
            color = 'Red'
        )
    )
]
plot_layout = go.Layout(
        title='Top 20 bi-grams from Covid 19 Tweets',
        yaxis_title='Count',
        xaxis_title='bi-gram',
        plot_bgcolor='rgba(0,0,0,0)'
    )
fig = go.Figure(data=plot_data, layout=plot_layout)
pyoff.iplot(fig)
# Tri-grams for Tweets

HQ_words = basic_clean(''.join(str(df['text'].tolist())))
trigram_HQ = (pd.Series(nltk.ngrams(HQ_words, 3)).value_counts())[:20]
trigram_HQ = pd.DataFrame(trigram_HQ)
trigram_HQ['idx'] = trigram_HQ.index
trigram_HQ['idx'] = trigram_HQ.apply(lambda x: '('+x['idx'][0]+', '+x['idx'][1]+', '+x['idx'][2]+')',axis=1)
plot_data = [
    go.Bar(
        x=trigram_HQ['idx'],
        y=trigram_HQ[0],
        marker = dict(
            color = 'Green'
        )
    )
]
plot_layout = go.Layout(
        title='Top 20 Tri-grams from Covid 19 Tweets',
        yaxis_title='Count',
        xaxis_title='Tri-gram',
        plot_bgcolor='rgba(0,0,0,0)'
    )
fig = go.Figure(data=plot_data, layout=plot_layout)
pyoff.iplot(fig)
test = pd.read_csv('../input/covid-19-nlp-text-classification/Corona_NLP_test.csv', encoding='latin-1')
test.head()
cleaned_title = []

for sentance in tqdm(test['OriginalTweet'].values):
    sentance = str(sentance)
    sentance = re.sub(r"http\S+", "", sentance)
    sentance = BeautifulSoup(sentance, 'lxml').get_text()
    sentance = cleaner(sentance)
    sentance = re.sub(r'[?|!|\'|"|#|+]', r'', sentance)
    sentance = re.sub('<.*?>','',sentance)
    sentance = re.sub(r'@\w+','',sentance)
    sentance = re.sub(r'#\w+','',sentance)
    sentance = re.sub(r'[0-9]+','',sentance)
    sentance = re.sub(r'[0-9]+','',sentance)
    sentance = re.sub("\S*\d\S*", "", sentance).strip()
    sentance = re.sub('[^A-Za-z]+', ' ', sentance)
    sentance = ' '.join(e.lower() for e in sentance.split() if e.lower() not in stop)
    cleaned_title.append(sentance.strip())
    
test['text'] = cleaned_title
test.head()
df['text'].replace('', np.nan, inplace=True)
df.dropna(subset=['text'], inplace=True)

df.head()
df.columns
train = df.copy()
train.drop(['UserName', 'ScreenName', 'Location', 'TweetAt', 'OriginalTweet', 'hashtags', 'mentions'], axis=1, inplace=True)
train.head()
X = train.text
def target(label):
    if label == 'Neutral': 
        return 0
    if label == 'Positive' or label=='Extremely Positive':
        return 1
    else:
        return -1
train['label'] = train['Sentiment'].apply(target)
train.head()
X = train['text']
y = train['label']
from sklearn.model_selection import train_test_split
X_Train, X_test, y_Train, y_test = train_test_split(X, y, test_size=0.1, random_state=42, stratify = y)

X_train, X_cross, y_train, y_cross = train_test_split(X_Train, y_Train, test_size=0.1, random_state=42, stratify = y_Train)
from sklearn.feature_extraction.text import TfidfVectorizer

tf_idf=TfidfVectorizer(use_idf=True,ngram_range=(1,2))

tf_idf.fit(X_train)
Train_TFIDF = tf_idf.transform(X_train)
CrossVal_TFIDF = tf_idf.transform(X_cross)
Test_TFIDF= tf_idf.transform(X_test)
# Logistic Regression
from sklearn.linear_model import LogisticRegression


c=[0.0001,0.001,0.01,0.1,1,10,100,1000]
Train_AUC_TFIDF = []
CrossVal_AUC_TFIDF = []
for i in c:
  logreg = LogisticRegression(C=i,penalty='l2')
  logreg.fit(Train_TFIDF, y_train)
  Train_y_pred =  logreg.predict_proba(Train_TFIDF)[0:,]
  Train_AUC_TFIDF.append(roc_auc_score(y_train ,Train_y_pred, multi_class='ovr'))
  CrossVal_y_pred =  logreg.predict_proba(CrossVal_TFIDF)[0:,]
  CrossVal_AUC_TFIDF.append(roc_auc_score(y_cross,CrossVal_y_pred, multi_class='ovr'))
C=[]
for i in range(len(c)):
  C.append(np.math.log(c[i]))

plt.plot(C, Train_AUC_TFIDF, label='Train AUC')
plt.scatter(C, Train_AUC_TFIDF)
plt.plot(C, CrossVal_AUC_TFIDF, label='CrossVal AUC')
plt.scatter(C, CrossVal_AUC_TFIDF)
plt.legend()
plt.xlabel("lambda : hyperparameter")
plt.ylabel("AUC")
plt.title("ERROR PLOTS")
plt.show()
optimal_inverse_lambda=c[CrossVal_AUC_TFIDF.index(max(CrossVal_AUC_TFIDF))]
print(pow(optimal_inverse_lambda,-1))
Classifier=LogisticRegression(C=optimal_inverse_lambda,penalty='l2')
Classifier.fit(Train_TFIDF, y_train)

auc_train_tfidf = roc_auc_score(y_train,Classifier.predict_proba(Train_TFIDF)[0:,], multi_class='ovr')
print ("AUC for Train set", auc_train_tfidf)

auc_test_tfidf = roc_auc_score(y_test,Classifier.predict_proba(Test_TFIDF)[0:,], multi_class='ovr')
print ("AUC for Test set",auc_test_tfidf)
y_pred = Classifier.predict(Test_TFIDF)
print('Confusion Matrix of Test Data')
Test_mat=confusion_matrix(y_test, y_pred)
print (Test_mat)

print('Accuracy Score on test: ', accuracy_score(y_test, y_pred))
from sklearn import metrics
print(metrics.classification_report(y_test, y_pred, target_names = ['Negative', 'Neutral', 'Positive']))
# LSTM
import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
numpy.random.seed(7)
from keras.layers import SpatialDropout1D
from keras.callbacks import EarlyStopping
from keras.preprocessing import text
MAX_NB_WORDS = 50000
MAX_SEQUENCE_LENGTH = 300
EMBEDDING_DIM = 100
tokenizer = text.Tokenizer(num_words=MAX_NB_WORDS, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
tokenizer.fit_on_texts(X.values)
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))
x = tokenizer.texts_to_sequences(X.values)
x = sequence.pad_sequences(x, maxlen=MAX_SEQUENCE_LENGTH)
print('Shape of data tensor:', x.shape)

X_train, X_test, Y_train, Y_test = train_test_split(x,y, test_size=0.3,stratify=y)
print(X_train.shape,Y_train.shape)
print(X_test.shape,Y_test.shape)
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

Y_train=np.array(Y_train)

label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(Y_train)
print(Y_train)

onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
print(onehot_encoded)
Y_train=onehot_encoded
Y_test=np.array(Y_test)

label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(Y_test)
#print(Y_test)

onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
print(onehot_encoded)
Y_test=onehot_encoded
model = Sequential()
model.add(Embedding(MAX_NB_WORDS, EMBEDDING_DIM, input_length=x.shape[1]))
model.add(SpatialDropout1D(0.2))
model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(3, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

epochs = 10
batch_size = 64

history = model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size,validation_split=0.1,callbacks=[EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001)])
accr = model.evaluate(X_test,Y_test)
print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0],accr[1]))
