import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
print(os.listdir('../input/movie-review-sentiment-analysis-kernels-only/'))
train_dir = pd.read_csv('../input/movie-review-sentiment-analysis-kernels-only/train.tsv.zip',sep='\t')
test_dir = pd.read_csv('../input/movie-review-sentiment-analysis-kernels-only/test.tsv.zip',sep='\t')
train_dir.head()
train_dir['Sentiment'].unique()
plt.style.use('seaborn')
plt.figure(figsize=(7,5))
sns.countplot(data=train_dir,x='Sentiment')
print(len(train_dir))
print(len(test_dir))
test_dir.head()
train_dir
train_dir.isna().sum()
X = train_dir.drop('Sentiment',axis=1)
X
y = train_dir['Sentiment']
y
import nltk
import re
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

corpus = []

for i in range(len(X)):
    text = re.sub('[^a-zA-Z]',' ',X['Phrase'][i])
    text = text.lower()
    text = text.split()
    
    text = [lemmatizer.lemmatize(word) for word in text if not word in nltk.corpus.stopwords.words('english')]
    text = ' '.join(text)
    corpus.append(text)
corpus
import keras 
from keras.utils import to_categorical
y = to_categorical(y)
y
ttest_dir = test_dir.drop('PhraseId',axis=1,inplace=True)
test_dir
test_corpus = []

for i in range(len(test_dir)):
    text = re.sub('[^a-zA-Z]',' ',test_dir['Phrase'][i])
    text = text.lower()
    text = text.split()
    
    text = [lemmatizer.lemmatize(word) for word in text if not word in nltk.corpus.stopwords.words('english')]
    text = ' '.join(text)
    test_corpus.append(text)
test_corpus
word2count = {}

for sentence in corpus:
    words = nltk.word_tokenize(sentence)
    
    for word in words:
        if word not in word2count.keys():
            word2count[word] = 1
        else:
            word2count[word] += 1
len(word2count)
import heapq
word_freq = heapq.nlargest(5000,word2count,key=word2count.get)
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding,Dense,LSTM,Dropout
from keras.preprocessing.text import one_hot
vocab_size = len(word_freq)
one_hot_train = []
for sentences in corpus:
    Z = one_hot(sentences,vocab_size)
    one_hot_train.append(Z)
one_hot_train[:5]
one_hot_test = []
for sentences in test_corpus:
    Z = one_hot(sentences,vocab_size)
    one_hot_test.append(Z)
one_hot_test[:2]
length = 20
train_embedded_sents = pad_sequences(one_hot_train,padding='pre',maxlen=length)
test_embedded_sents = pad_sequences(one_hot_test,padding='pre',maxlen=length)
train_embedded_sents[:2]
test_embedded_sents[:2]
embedding_feature_vectors = 40
model = Sequential()
model.add(Embedding(vocab_size,embedding_feature_vectors,input_length=length))
model.add(Dropout(0.2))
model.add(LSTM(100))
model.add(Dropout(0.3))
model.add(Dense(5,activation='softmax'))
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
X_final = np.asarray(train_embedded_sents)
y_final = np.asarray(y)
from sklearn.model_selection import train_test_split
X_train,X_valid,y_train,y_valid = train_test_split(X_final,y_final,test_size=0.2)
len(X_train)
len(X_valid)
history = model.fit(X_train,y_train,validation_data=(X_valid,y_valid),epochs=20,batch_size=128)
plt.style.use('dark_background')
fig, ax = plt.subplots(2, 2, figsize=(10, 10))
sns.lineplot(x=np.arange(1, 21), y=history.history.get('loss'), ax=ax[0, 0])
sns.lineplot(x=np.arange(1, 21), y=history.history.get('accuracy'), ax=ax[0, 1])
sns.lineplot(x=np.arange(1, 21), y=history.history.get('val_loss'), ax=ax[1, 0])
sns.lineplot(x=np.arange(1, 21), y=history.history.get('val_accuracy'), ax=ax[1, 1])
ax[0, 0].set_title('Training Loss vs Epochs')
ax[0, 1].set_title('Training Accuracy vs Epochs')
ax[1, 0].set_title('Validation Loss vs Epochs')
ax[1, 1].set_title('Validation Accuracy vs Epochs')
plt.show()
test = np.asarray(test_embedded_sents)
test
sub = pd.read_csv('../input/movie-review-sentiment-analysis-kernels-only/sampleSubmission.csv')
sub['labels'] = model.predict_classes(test,batch_size=128)
sub
from sklearn.metrics import accuracy_score,confusion_matrix
score = accuracy_score(sub['Sentiment'],sub['labels'])
print(score)
