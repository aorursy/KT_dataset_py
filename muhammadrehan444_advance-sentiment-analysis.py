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
dataset = pd.read_csv('/kaggle/input/sentiment140/training.1600000.processed.noemoticon.csv', encoding="ISO-8859-1", names=["sentiment", "ids", "date", "flag", "user", "text"])

decode = {0 : 'Negative', 2: 'Neutral', 4: 'Positive'}
def deco(x):
    return decode[int(x)]
%%time 
dataset['sentiment'] = dataset['sentiment'].apply(lambda y : deco(y))
dataset.head()
from collections import Counter
count = Counter(dataset['sentiment'])
count
import matplotlib.pyplot as plt
plt.bar(count.keys(), count.values())
from  nltk.stem import SnowballStemmer
from nltk.corpus import stopwords

stop_words = stopwords.words('english')
stemmer= SnowballStemmer('english')
import re
TEXT_CLEANING_RE = "@\S+|https?:\S+|http?:\S|[^A-Za-z0-9]+"
def preprocess(text, stem =False):
    text = re.sub(TEXT_CLEANING_RE, ' ', str(text).lower()).strip()
    tokens = []
    for token in text.split():
        if token not in stop_words:
            if stem:
                tokens.append(stemmer.stem(token))
            else: 
                tokens.append(token)
    return " ".join(tokens)
dataset['text']= dataset['text'].apply(lambda x : preprocess(x))
dataset['text']
from sklearn.model_selection import train_test_split

train , test = train_test_split(dataset, test_size=0.2, random_state= 42)
print("TRAIN size:", len(train))
print("TEST size:", len(test))

%%time
documents = [d.split() for d in train['text']]
documents
import gensim

w2v_model = gensim.models.word2vec.Word2Vec(size=300, 
                                            window=7, 
                                            min_count=10, 
                                            workers=8)
w2v_model.build_vocab(documents)
words = w2v_model.wv.vocab.keys()
vocab_size = len(words)
print("Vocab size", vocab_size)

%%time
w2v_model.train(documents, total_examples=len(documents), epochs=32)

w2v_model.most_similar("love")

from keras.preprocessing.text import Tokenizer
tokeni = Tokenizer()
tokeni.fit_on_texts(train['text'])

vocab_size = len(tokeni.word_index) + 1
print("Total words", vocab_size)

from keras.preprocessing.sequence import pad_sequences
x_train = pad_sequences(tokeni.texts_to_sequences(train['text']), maxlen= 300)
x_test = pad_sequences(tokeni.texts_to_sequences(test['text']), maxlen= 300)

x_train
labels = train['sentiment'].unique()
labels
labels=labels.tolist()
labels.append("NEUTRAL")
labels
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
encoder.fit(train['sentiment'].tolist())
y_train = encoder.transform(train['sentiment'].tolist())
y_test = encoder.transform(test['sentiment'].tolist())
print("y_train",y_train.shape)
print("y_test",y_test.shape)


y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1)

print("y_train",y_train.shape)
print("y_test",y_test.shape)

print("x_train", x_train.shape)
print("y_train", y_train.shape)
print()
print("x_test", x_test.shape)
print("y_test", y_test.shape)

y_train[:10]

embedding_matrics = np.zeros((vocab_size, 300))
for word, i in tokeni.word_index.items():
    if word in w2v_model.wv:
        embedding_matrics[i]= w2v_model.wv[word]
print(embedding_matrics.shape)

from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout, Embedding, Flatten, Conv1D, MaxPooling1D, LSTM
from keras import utils
from keras.callbacks import ReduceLROnPlateau, EarlyStopping

embedding_layer = Embedding(vocab_size, 300, weights=[embedding_matrics], input_length=300, trainable=False)

model = Sequential()
model.add(embedding_layer)
model.add(Dropout(0.5))
model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))

model.summary()
model.compile(loss='binary_crossentropy',
              optimizer="adam",
              metrics=['accuracy'])
callbacks = [ ReduceLROnPlateau(monitor='val_loss', patience=5, cooldown=0),
              EarlyStopping(monitor='val_acc', min_delta=1e-4, patience=5)]

%%time
history = model.fit(x_train, y_train,
                    batch_size=1024,
                    epochs=1,
                    validation_split=0.1,
                    verbose=1,
                    callbacks=callbacks)
score = model.evaluate(x_test, y_test, batch_size=1)
print("ACCURACY:",score[1])
print("LOSS:",score[0])

import time
SENTIMENT_THRESHOLDS = (0.4, 0.7)

def decode_sentiment(score, include_neutral=True):
    if include_neutral:        
        label = "NEUTRAL"
        if score <= SENTIMENT_THRESHOLDS[0]:
            label = "NEGATIVE"
        elif score >= SENTIMENT_THRESHOLDS[1]:
            label = "POSITIVE"

        return label
    else:
        return "NEGATIVE" if score < 0.5 else "POSITIVE"

def predict(text, include_neutral=True):
    start_at = time.time()
    # Tokenize text
    x_test = pad_sequences(tokeni.texts_to_sequences([text]), maxlen=300)
    # Predict
    score = model.predict([x_test])[0]
    # Decode sentiment
    label = decode_sentiment(score, include_neutral=include_neutral)

    return {"label": label, "score": float(score),
       "elapsed_time": time.time()-start_at}  
predict("I love the music")

predict("I love her")

