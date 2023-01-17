import pandas as pd

import numpy as np

%matplotlib inline

import matplotlib.pyplot as plt

plt.style.use('ggplot')

import seaborn as sns

import re

import string

import nltk

from nltk.corpus import stopwords

from nltk.stem import SnowballStemmer,WordNetLemmatizer

from string import punctuation

from wordcloud import WordCloud

from nltk.tokenize import TweetTokenizer

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn import preprocessing, decomposition, model_selection, metrics, pipeline

from sklearn.linear_model import LogisticRegressionCV

from sklearn.svm import SVC

from sklearn.naive_bayes import MultinomialNB

from sklearn.decomposition import TruncatedSVD

import xgboost as xgb

from sklearn.model_selection import train_test_split,GridSearchCV,StratifiedKFold

from sklearn.metrics import f1_score,accuracy_score,precision_score,recall_score,confusion_matrix

import tensorflow as tf

from tensorflow import keras

from keras.callbacks import EarlyStopping,ReduceLROnPlateau

from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from keras.layers import Dense,Embedding,Bidirectional,Dropout,SpatialDropout1D,GlobalMaxPool1D,LSTM,BatchNormalization,Conv1D,MaxPool1D

from keras.models import Sequential,load_model

from keras.optimizers import Adam,RMSprop

from keras import regularizers
train = pd.read_csv("../input/movie-review-sentiment-analysis-kernels-only/train.tsv.zip",sep="\t")

test = pd.read_csv("../input/movie-review-sentiment-analysis-kernels-only/test.tsv.zip",sep="\t")

sub = pd.read_csv("../input/movie-review-sentiment-analysis-kernels-only/sampleSubmission.csv")
train.head()
#Count Of Tweets

sns.countplot(train['Sentiment'],palette='rocket_r')

plt.title("No of Tweet Sentiments")
#Phrases Per Sentence

fig,ax = plt.subplots(figsize=(15,5))

plt.subplot(1,2,1)

sns.distplot(train.groupby('SentenceId')['PhraseId'].count())

plt.title("Avg Phrases Per Sentence in Train")

plt.subplot(1,2,2)

sns.distplot(test.groupby('SentenceId')['PhraseId'].count())

plt.title("Avg Phrases Per Sentence in Test")



print("Avg Phrases Per Sentence in Train: ",round(train.groupby('SentenceId')['PhraseId'].count().mean()))

print("Avg Phrases Per Sentence in Test: ",round(test.groupby('SentenceId')['PhraseId'].count().mean()))
#No of Words in Phrases

fig,ax = plt.subplots(figsize=(22,5))

fig.suptitle("Avg Words In Phrases",fontsize=16)



plt.subplot(1,5,1)

sns.distplot(train[train['Sentiment']==0]['Phrase'].str.split().apply(lambda x:len(x)))

plt.title("Sentiment 0")

print("Avg Words in Phrases with Sentiment 0: ",round(train[train['Sentiment']==0]['Phrase'].str.split().apply(lambda x:len(x)).mean()))



plt.subplot(1,5,2)

sns.distplot(train[train['Sentiment']==1]['Phrase'].str.split().apply(lambda x:len(x)))

plt.title("Sentiment 1")

print("Avg Words in Phrases with Sentiment 1: ",round(train[train['Sentiment']==1]['Phrase'].str.split().apply(lambda x:len(x)).mean()))



plt.subplot(1,5,3)

sns.distplot(train[train['Sentiment']==2]['Phrase'].str.split().apply(lambda x:len(x)))

plt.title("Sentiment 2")

print("Avg Words in Phrases with Sentiment 2: ",round(train[train['Sentiment']==2]['Phrase'].str.split().apply(lambda x:len(x)).mean()))



plt.subplot(1,5,4)

sns.distplot(train[train['Sentiment']==3]['Phrase'].str.split().apply(lambda x:len(x)))

plt.title("Sentiment 3")

print("Avg Words in Phrases with Sentiment 3: ",round(train[train['Sentiment']==3]['Phrase'].str.split().apply(lambda x:len(x)).mean()))



plt.subplot(1,5,5)

sns.distplot(train[train['Sentiment']==4]['Phrase'].str.split().apply(lambda x:len(x)))

plt.title("Sentiment 4")

print("Avg Words in Phrases with Sentiment 4: ",round(train[train['Sentiment']==4]['Phrase'].str.split().apply(lambda x:len(x)).mean()))
#Avg Characters in Phrases

fig,ax = plt.subplots(figsize=(22,5))

fig.suptitle("Avg Characters In Phrases",fontsize=16)



plt.subplot(1,5,1)

sns.distplot(train[train['Sentiment']==0]['Phrase'].str.len())

plt.title("Sentiment 0")

print("Avg Characters in Phrases with Sentiment 0: ",round(train[train['Sentiment']==0]['Phrase'].str.len().mean()))



plt.subplot(1,5,2)

sns.distplot(train[train['Sentiment']==1]['Phrase'].str.len())

plt.title("Sentiment 1")

print("Avg Characters in Phrases with Sentiment 1: ",round(train[train['Sentiment']==1]['Phrase'].str.len().mean()))



plt.subplot(1,5,3)

sns.distplot(train[train['Sentiment']==2]['Phrase'].str.len())

plt.title("Sentiment 2")

print("Avg Characters in Phrases with Sentiment 2: ",round(train[train['Sentiment']==2]['Phrase'].str.len().mean()))



plt.subplot(1,5,4)

sns.distplot(train[train['Sentiment']==3]['Phrase'].str.len())

plt.title("Sentiment 3")

print("Avg Characters in Phrases with Sentiment 3: ",round(train[train['Sentiment']==3]['Phrase'].str.len().mean()))



plt.subplot(1,5,5)

sns.distplot(train[train['Sentiment']==4]['Phrase'].str.len())

plt.title("Sentiment 4")

print("Avg Characters in Phrases with Sentiment 4: ",round(train[train['Sentiment']==4]['Phrase'].str.len().mean()))

#WordCloud

fig,ax = plt.subplots(figsize=(20,40))

plt.axis('off')



plt.subplot(5,1,1)

text = " ".join(train[train['Sentiment']==0]['Phrase'])

wordcloud = WordCloud(max_font_size = None,background_color='white',width=1000,height=1000).generate(text)

plt.title("WordCloud for Sentiment 0")

plt.axis('off')

plt.imshow(wordcloud)



plt.subplot(5,1,2)

text = " ".join(train[train['Sentiment']==1]['Phrase'])

wordcloud = WordCloud(max_font_size = None,background_color='white',width=1000,height=1000).generate(text)

plt.title("WordCloud for Sentiment 1")

plt.axis('off')

plt.imshow(wordcloud)



plt.subplot(5,1,3)

text = " ".join(train[train['Sentiment']==2]['Phrase'])

wordcloud = WordCloud(max_font_size = None,background_color='white',width=1000,height=1000).generate(text)

plt.title("WordCloud for Sentiment 2")

plt.axis('off')

plt.imshow(wordcloud)



plt.subplot(5,1,4)

text = " ".join(train[train['Sentiment']==3]['Phrase'])

wordcloud = WordCloud(max_font_size = None,background_color='white',width=1000,height=1000).generate(text)

plt.title("WordCloud for Sentiment 3")

plt.axis('off')

plt.imshow(wordcloud)



plt.subplot(5,1,5)

text = " ".join(train[train['Sentiment']==4]['Phrase'])

wordcloud = WordCloud(max_font_size = None,background_color='white',width=1000,height=1000).generate(text)

plt.title("WordCloud for Sentiment 4")

plt.axis('off')

plt.imshow(wordcloud)

stemmer = SnowballStemmer('english',ignore_stopwords=True)

lemmatizer = WordNetLemmatizer()

def clean(text):

    sentence=[]

    for word in text.split():

        word = re.sub('[^a-zA-Z]','',word)

        word = word.lower()

        word = lemmatizer.lemmatize(word)

        word = word.strip()

        sentence.append(word)

    return " ".join(sentence)



train['Phrase'] = train['Phrase'].apply(lambda x:clean(x))

test['Phrase'] = test['Phrase'].apply(lambda x:clean(x))
x_train,x_valid,y_train,y_valid = train_test_split(train['Phrase'],train['Sentiment'],test_size=0.2,random_state=42)
#Tokenize

vocab_size=20000

embedding_dim=200

max_length=50

trunc_type="post"

pad_type="post"

oov_tok="<OOV>"

epochs=10

batch_size=128



tokenizer = Tokenizer(num_words=vocab_size,oov_token=oov_tok)

tokenizer.fit_on_texts(list(x_train)+list(x_valid))

word_index = tokenizer.word_index



train_seq = tokenizer.texts_to_sequences(x_train)

train_pad = pad_sequences(train_seq,maxlen=max_length,truncating = trunc_type,padding=pad_type)



val_seq = tokenizer.texts_to_sequences(x_valid)

val_pad = pad_sequences(val_seq,maxlen=max_length,truncating = trunc_type,padding=pad_type)
len(word_index)
#Glove Embeddings

embeddings_index={}

with open("../input/glove6b/glove.6B.200d.txt",'r',encoding='utf-8') as f:

    for line in f:

        values = line.split()

        word = values[0]

        coefs = np.asarray(values[1:], dtype='float32')

        embeddings_index[word] = coefs

        

embeddings_matrix = np.zeros((len(word_index)+1, embedding_dim))

for word, i in word_index.items():

    embedding_vector = embeddings_index.get(word)

    if embedding_vector is not None:

        embeddings_matrix[i] = embedding_vector
#Model



model = Sequential()

model.add(Embedding(len(word_index)+1,embedding_dim,input_length=max_length,weights=[embeddings_matrix]))

model.add(SpatialDropout1D(0.4))



model.add(Conv1D(128,3,activation='relu',padding='same'))

model.add(MaxPool1D(2))



model.add(Conv1D(64,3,activation='relu',padding='same'))

model.add(MaxPool1D(2))



model.add(Bidirectional(LSTM(64,recurrent_dropout=0.5,dropout=0.5,return_sequences=True)))

model.add(Bidirectional(LSTM(64,recurrent_dropout=0.5,dropout=0.5,return_sequences=True)))



model.add(GlobalMaxPool1D())



model.add(Dense(64,activation='relu'))

model.add(BatchNormalization())

model.add(Dropout(0.3))



model.add(Dense(32,activation='relu'))

model.add(BatchNormalization())

model.add(Dropout(0.3))



model.add(Dense(5,activation='softmax'))



model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['acc'])

model.summary()
#Callbacks

earlystop = EarlyStopping(monitor='val_loss',patience=2,verbose=1)

learning_reduce = ReduceLROnPlateau(patience=1,monitor="val_acc",verbose=1,min_lr=0.00001,factor=0.5,cooldown=1)

callbacks = [earlystop,learning_reduce]
history = model.fit(train_pad,y_train,epochs=epochs,validation_data=(val_pad,y_valid),callbacks=callbacks,

                    batch_size=batch_size)
#Plot

def plot_graphs(history, string):

    plt.plot(history.history[string])

    plt.plot(history.history["val_"+string])

    plt.xlabel("Epochs")

    plt.ylabel(string)

    plt.legend([string,"val_"+string])

    plt.show()

plot_graphs(history,'acc')

plot_graphs(history,'loss')
tweet_tokenizer = TweetTokenizer()

tfidf = TfidfVectorizer(strip_accents=None,lowercase=False,preprocessor=None,ngram_range=(1,3),

                        tokenizer=tweet_tokenizer.tokenize,use_idf=True,norm='l2',smooth_idf=True)

tfidf.fit(list(x_train.values) + list(x_valid.values))

xtrain_tfv = tfidf.transform(x_train)

xvalid_tfv = tfidf.transform(x_valid)



scl = preprocessing.StandardScaler(with_mean=False)

xtrain_tfv_std = scl.fit_transform(xtrain_tfv)

xvalid_tfv_std = scl.transform(xvalid_tfv)
logistic = LogisticRegressionCV(cv=3,scoring='accuracy',random_state=42,n_jobs=-1,verbose=3)

logistic.fit(xtrain_tfv_std,y_train)

logistic_accuracy = logistic.score(xvalid_tfv_std,y_valid)

print("Accuracy:",logistic_accuracy)
svc = SVC(C=0.1,random_state=42,verbose=2)

svc.fit(xtrain_tfv_std,y_train)

svc_accuracy = svc.score(xvalid_tfv_std,y_valid)

print("Accuracy:",svc_accuracy)
naive = MultinomialNB()

naive.fit(xtrain_tfv_std,y_train)

naive_accuracy = naive.score(xvalid_tfv_std,y_valid)

print("Accuracy:",naive_accuracy)
xgboost = xgb.XGBClassifier(max_depth=7, n_estimators=200, colsample_bytree=0.8, 

                            subsample=0.8, nthread=10, learning_rate=0.1,verbose=2)

xgboost.fit(xtrain_tfv_std,y_train)

xg_accuracy = xgboost.score(xvalid_tfv_std,y_valid)

print("Accuracy:",xg_accuracy)
test_sequences = tokenizer.texts_to_sequences(test['Phrase'])

test_pad = pad_sequences(test_sequences,maxlen=max_length,truncating=trunc_type,padding=pad_type)
# CNN+LSTM

ypred = model.predict_classes(test_pad,verbose=1)

sub['Sentiment'] = ypred

sub.to_csv("submission.csv", index=False, header=True)