# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

%matplotlib inline

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import nltk

from nltk.corpus import stopwords

import re

import joblib

from keras.preprocessing.text import Tokenizer

import gensim

from keras.preprocessing.sequence import pad_sequences

from sklearn.preprocessing import LabelEncoder

from keras.layers import Embedding

from keras.models import Sequential

from keras.layers import Dense,LSTM,Dropout

from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
df = pd.read_csv("/kaggle/input/sentiment140/training.1600000.processed.noemoticon.csv",encoding='latin-1',header=None)
df.head()
columns=['target','ids','date','flag','user','text']

df.columns=columns
df.head()
df.target.replace({0:'Negative',2:'Neutral',4:'Positive'},inplace=True)
df.head()
stop_words=set(stopwords.words('english'))

stop_words.remove('not')


corpus=[]

for i in range(0,len(df)):

    review=re.sub('@\S+|https?:\S+|http?:\S|[^A-Za-z0-9]+',' ',df['text'][i])

    review=review.lower()

    review=review.split()

    review=[word for word in review if not word in stop_words]

    review=' '.join(review)

    corpus.append(review)
df.text=corpus
df.head()
from sklearn.model_selection import train_test_split

train_df,test_df=train_test_split(df,test_size=0.20,random_state=123)
train_df.head()
test_df.head()
documents = [text.split() for text in train_df.text]
w2v_model = gensim.models.word2vec.Word2Vec(size=300, 

                                            window=7, 

                                            min_count=10, 

                                            workers=8)
w2v_model.build_vocab(documents)
words = w2v_model.wv.vocab.keys()

vocab_size = len(words)

print("Vocab size", vocab_size)
w2v_model.train(documents, total_examples=len(documents), epochs=30)
w2v_model.wv.most_similar("good")
w2v_model.wv.most_similar("hate")
w2v_model.wv.most_similar("great")
tokenizer = Tokenizer()

tokenizer.fit_on_texts(train_df.text)
tokenizer.word_index
vocab_size=len(tokenizer.word_index)+1

vocab_size
X_train = pad_sequences(tokenizer.texts_to_sequences(train_df.text), maxlen=300)

X_train
X_test = pad_sequences(tokenizer.texts_to_sequences(test_df.text), maxlen=300)

X_test
y_train=train_df.target

y_train.head()
y_test=test_df.target

y_test.head()
labelencoder = LabelEncoder()

y_train = labelencoder.fit_transform(y_train)

y_test=labelencoder.fit_transform(y_test)
y_train.shape
y_test.shape
embedding_matrix = np.zeros((vocab_size, 300))

for word, i in tokenizer.word_index.items():

  if word in w2v_model.wv:

    embedding_matrix[i] = w2v_model.wv[word]

print(embedding_matrix.shape)
embedding_layer = Embedding(vocab_size, 300, weights=[embedding_matrix], input_length=300, trainable=False)
model = Sequential()

model.add(embedding_layer)

model.add(Dropout(0.5))

model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))

model.add(Dense(1, activation='sigmoid'))

model.summary()
model.compile(loss='binary_crossentropy',

              optimizer="adam",

              metrics=['accuracy'])
model_history=model.fit(X_train, y_train,batch_size=1024,epochs=15,validation_split=0.1,verbose=1)
acc = model_history.history['accuracy']

val_acc = model_history.history['val_accuracy']

loss = model_history.history['loss']

val_loss = model_history.history['val_loss']

epochs=range(len(acc))
plt.plot(epochs,acc,label='Trainin_acc',color='blue')

plt.plot(epochs,val_acc,label='Validation_acc',color='red')

plt.legend()

plt.title("Training and Validation Accuracy")
plt.plot(epochs,loss,label='Training_loss',color='blue')

plt.plot(epochs,val_loss,label='Validation_loss',color='red')

plt.legend()

plt.title("Training and Validation loss")
def preprocess(text):

    review=re.sub('@\S+|https?:\S+|http?:\S|[^A-Za-z0-9]+',' ',text)

    review=review.lower()

    review=review.split()

    review=[word for word in review if not word in stop_words]

    print(review)

    review=pad_sequences(tokenizer.texts_to_sequences([review]), maxlen=300)

    return review
def prediction(review):

    review=preprocess(review)

    score=model.predict(review)

    score=score[0]

    if score<0.4:

        print("Negative")

    elif score>0.4 and score<0.6:

        print("Neutral")

    else:

        print("Positive")

    print(score)

prediction("the food is not bad")
prediction("the film was horrible")
scores = model.predict(X_test, verbose=1, batch_size=1024)
scores
y_pred=np.where(scores>0.5,1,0)
y_pred
y_test


cm=confusion_matrix(y_pred,y_test)

print(cm)
print(accuracy_score(y_pred,y_test))
print(classification_report(y_test, y_pred))
joblib.dump(w2v_model,'word2vec.pkl')

joblib.dump(tokenizer,'tokenizer.pkl')

joblib.dump(model,'final_model.pkl')