# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np
import numpy # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import string
from textblob import TextBlob
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize

from nltk import FreqDist
from nltk.stem import SnowballStemmer,WordNetLemmatizer
stemmer=SnowballStemmer('english')
lemma=WordNetLemmatizer()
from string import punctuation
import re


import gc
from keras.preprocessing import sequence,text
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Dense,Dropout,Embedding,LSTM,Conv1D,GlobalMaxPooling1D,Flatten,MaxPooling1D,GRU,SpatialDropout1D,Bidirectional
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report,f1_score

import os
print(os.listdir("../input"))

import warnings
warnings.filterwarnings("ignore")
# Any results you write to the current directory are saved as output.
train = pd.read_csv("../input/movie-review-sentiment-analysis-kernels-only/train.tsv",sep="\t")
test = pd.read_csv("../input/movie-review-sentiment-analysis-kernels-only/test.tsv",sep="\t")
sub=pd.read_csv('../input/movie-review-sentiment-analysis-kernels-only/sampleSubmission.csv')
train.head()
test.head()
def clean_review(review_col):
    review_corpus=[]
    for i in range(0,len(review_col)):
        review=str(review_col[i])
        review=re.sub('[^a-zA-Z]',' ',review)
        #review=[stemmer.stem(w) for w in word_tokenize(str(review).lower())]
        review=[lemma.lemmatize(w) for w in word_tokenize(str(review).lower())]
        review=' '.join(review)
        review_corpus.append(review)
    return review_corpus
train['csen']=clean_review(train.Phrase.values)
train.head()
test['csen']=clean_review(test.Phrase.values)
test.head()
train['word'] = train['csen'].apply(lambda x: len(TextBlob(x).words))
test['word'] = test['csen'].apply(lambda x: len(TextBlob(x).words))
train = train[train['word'] >= 1]
train = train.reset_index(drop=True)
train.head()
test.head()
s_a = train.groupby('Sentiment')['word'].describe().reset_index()
s_a
### plot packages
import plotly.offline as py
py.init_notebook_mode(connected=True)
from plotly.offline import init_notebook_mode, iplot
import plotly.figure_factory as ff
import matplotlib as plt
import plotly.graph_objs as go
import plotly.tools as tls
%matplotlib inline
import cufflinks as cf
cf.go_offline()
trace1 = go.Bar(
    x=s_a['Sentiment'],
    y=s_a['count'] 
)

data = [trace1]
layout = go.Layout(
    title = 'Sentiment_Count'
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='grouped-bar')
trace1 = go.Bar(
    x=s_a['Sentiment'],
    y=s_a['min'],
    name='Min Sentence length'
)

trace2 = go.Bar(
    x=s_a['Sentiment'],
    y=s_a['mean'],
    name='Average Sentence length'
)

trace3 = go.Bar(
    x=s_a['Sentiment'],
    y=s_a['max'],
    name='Max Sentence length'
)

data = [trace1, trace2,trace3]
layout = go.Layout(
    barmode='group',
    title ='Sentence length anlysis sentiment wise'
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='grouped-bar')
count = lambda l1, l2: len(list(filter(lambda c: c in l2, l1)))
train['pun_count'] = train['Phrase'].apply(lambda x: count(x, string.punctuation)) 
test['pun_count'] = test['Phrase'].apply(lambda x: count(x, string.punctuation))
pun_count = train.groupby('Sentiment')['pun_count'].sum().reset_index()
trace1 = go.Bar(
    x=pun_count['Sentiment'],
    y=pun_count['pun_count'],
    marker=dict(
        color='rgba(222,45,38,0.8)',
    )
)

data = [trace1]
layout = go.Layout(
    title = 'Sentiment wise Punctuation Count'
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='grouped-bar')
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from os import path
import warnings 
from os import path
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os

from wordcloud import WordCloud, STOPWORDS
warnings.filterwarnings('ignore')
stopwords = set(STOPWORDS)
def plot_wordcloud(text, mask=None, max_words=400, max_font_size=120, figure_size=(24.0,16.0), 
                   title = None, title_size=40, image_color=False):
    wordcloud = WordCloud(background_color='white',
                    stopwords = stopwords,
                    max_words = 200,
                    max_font_size = 200, 
                    random_state = 42,
                    mask = mask,contour_width=2)
    wordcloud.generate(text)
    
    plt.figure(figsize=figure_size)
    if image_color:
        image_colors = ImageColorGenerator(mask);
        plt.imshow(wordcloud.recolor(color_func=image_colors), interpolation="bilinear");
        plt.title(title, fontdict={'size': title_size,  
                                  'verticalalignment': 'bottom'})
    else:
        plt.imshow(wordcloud);
        plt.title(title, fontdict={'size': title_size, 'color': 'green', 
                                  'verticalalignment': 'bottom'})
    plt.axis('off');
    plt.tight_layout()  
comments_text = str(train[train['Sentiment'] == 0]['csen'].tolist())
comments_mask = np.array(Image.open('../input/mask-imges/use.png'))
plot_wordcloud(comments_text, comments_mask, max_words=400, max_font_size=120,title = 'Word Cloud Plot for Sentiment 0', title_size=15)
comments_text = str(train[train['Sentiment'] == 1]['csen'].tolist())
comments_mask = np.array(Image.open('../input/mask-imges/coment.png'))
plot_wordcloud(comments_text, comments_mask, max_words=400, max_font_size=120, 
               title = 'Word Cloud Plot for Sentiment 1', title_size=15,image_color=True)
text = str(train[train['Sentiment'] == 2]['csen'].tolist())
text = text.lower()
wordcloud = WordCloud(background_color="white", height=2700, width=3600).generate(text)
plt.figure( figsize=(16,10) )
plt.title('Word Cloud Plot for Sentiment 2',fontsize = 15)
plt.imshow(wordcloud.recolor(colormap=plt.get_cmap('Set2')), interpolation='bilinear')
plt.axis("off")
text = str(train[train['Sentiment'] == 3]['csen'].tolist())
text = text.lower()
wordcloud = WordCloud(max_words=50).generate(text)

labels = list(wordcloud.words_.keys())
values = list(wordcloud.words_.values())
trace = go.Pie(labels=labels, values=values,textinfo='value', hoverinfo='label+value',textposition = 'inside')
layout = go.Layout(
    title = 'Word Pie Plot for Sentiment 3'
)
data = [trace]
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='basic_pie_chart')

text = str(train[train['Sentiment'] == 4]['csen'].tolist())
text = text.lower()
wordcloud = WordCloud(max_words=50).generate(text)

labels = list(wordcloud.words_.keys())
values = list(wordcloud.words_.values())
trace1 = go.Bar(
    x=labels,
    y=values
    
)

layout = go.Layout(
    title = 'Word Bar Chart for Sentiment 4'
)
data = [trace1]
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='basic_pie_chart')

from sklearn.cluster import KMeans
import numpy as np

x = np.array(train['word'])
km = KMeans(n_clusters = 4)
km.fit(x.reshape(-1,1))  
train['cluster'] = list(km.labels_)
y = np.array(test['word'])
km = KMeans(n_clusters = 4)
km.fit(y.reshape(-1,1))  
test['cluster'] = list(km.labels_)
cluster = train.groupby(['Sentiment','cluster'])['word'].describe().reset_index()
cluster
pun_count = train.groupby(['Sentiment','cluster'])['pun_count'].describe().reset_index()
pun_count
train.groupby(['Sentiment','cluster'])['word'].count().unstack().plot(kind='bar', stacked=False)
train.groupby(['Sentiment','cluster'])['word'].mean().unstack().plot(kind='bar', stacked=False)
train.groupby(['Sentiment','cluster'])['word'].min().unstack().plot(kind='bar', stacked=False)
train.groupby(['Sentiment','cluster'])['word'].max().unstack().plot(kind='bar', stacked=False)
gc.collect()
train_text=train.filter(['csen','cluster'])
test_text=test.filter(['csen','cluster'])
target=train.Sentiment.values
y=to_categorical(target)
print(train_text.shape,target.shape,y.shape)
X_train_text,X_val_text,y_train,y_val=train_test_split(train_text,y,test_size=0.2,stratify=y,random_state=123)
print(X_train_text.shape,y_train.shape)
print(X_val_text.shape,y_val.shape)
all_words=' '.join(X_train_text.csen.values)
all_words=word_tokenize(all_words)
dist=FreqDist(all_words)
num_unique_word=len(dist)
num_unique_word
r_len=[]
for text in X_train_text.csen.values:
    word=word_tokenize(text)
    l=len(word)
    r_len.append(l)
    
MAX_REVIEW_LEN=np.max(r_len)
MAX_REVIEW_LEN
max_features = num_unique_word
max_words = MAX_REVIEW_LEN
batch_size = 128
epochs = 3
num_classes=5
tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(X_train_text.csen.values))
X_train = tokenizer.texts_to_sequences(X_train_text.csen.values)
X_val = tokenizer.texts_to_sequences(X_val_text.csen.values)
X_test = tokenizer.texts_to_sequences(test.csen.values)
X_train = sequence.pad_sequences(X_train, maxlen=max_words)
X_val = sequence.pad_sequences(X_val, maxlen=max_words)
X_test = sequence.pad_sequences(X_test, maxlen=max_words)
print(X_train.shape,X_val.shape,X_test.shape)
X_train = numpy.insert(X_train,48,numpy.array([X_train_text.cluster.values]),axis=1)
X_val = numpy.insert(X_val,48,numpy.array([X_val_text.cluster.values]),axis=1)
X_test = numpy.insert(X_test,48,numpy.array([test.cluster.values]),axis=1)
print(X_train.shape,X_val.shape,X_test.shape)
gc.collect()
model=Sequential()
model.add(Embedding(max_features,100,mask_zero=True))
model.add(LSTM(64,dropout=0.4, recurrent_dropout=0.4,return_sequences=True))
model.add(LSTM(32,dropout=0.5, recurrent_dropout=0.5,return_sequences=False))
model.add(Dense(num_classes,activation='softmax'))
model.compile(loss='categorical_crossentropy',optimizer=Adam(lr=0.001),metrics=['accuracy'])
model.summary()
%%time
history = model.fit(X_train, y_train, validation_data=(X_val, y_val),epochs=epochs, batch_size=batch_size,verbose=1)
y_pred1=model.predict_classes(X_test,verbose=1)

sub.Sentiment=y_pred1
sub.to_csv('sub.csv',index=False)
sub.head()
unique, counts = numpy.unique(y_pred1, return_counts=True)
dict(zip(unique, counts))