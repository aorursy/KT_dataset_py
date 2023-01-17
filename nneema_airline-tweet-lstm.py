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
df = pd.read_csv("/kaggle/input/twitter-airline-sentiment/Tweets.csv")
df.head()
import nltk

from nltk.corpus import stopwords

from nltk.tokenize import TreebankWordTokenizer
def tokenizer(text):

    stop_words = set(stopwords.words('english')) 

    stop_words.add("@")

    tokenizer_obj = TreebankWordTokenizer()

    word_list = tokenizer_obj.tokenize(text)

    filtered_words = [w.lower() for w in word_list if w not in stop_words]

    snow_stemmer = nltk.stem.SnowballStemmer("english")

    stemmed = [snow_stemmer.stem(w) for w in filtered_words]

    return " ".join(stemmed)

    
df["tokenized_text"] = df["text"].apply(tokenizer)
df["tokenized_text"].head()
df["text"].head()
from keras.preprocessing.text import Tokenizer
t = Tokenizer()

t.fit_on_texts(df["tokenized_text"])
# summarize what was learned

# print(t.word_counts)

# print(t.document_count)

# print(t.word_index)

# print(t.word_docs)
vocab_size=len(t.word_index)+1 

print(vocab_size)
X=t.texts_to_sequences(df['tokenized_text'].values)

    
max_len=max(len(row) for row in df['tokenized_text'].values)

print(max_len)
from tensorflow.keras.preprocessing.sequence import pad_sequences

X=pad_sequences(X, maxlen=max_len)

print(X.shape)
y=pd.get_dummies(df['airline_sentiment']).values

print(y[0])
from keras.layers import Dense, Dropout, LSTM, Embedding

from keras.models import Sequential

from keras.regularizers import l2
model=Sequential()

model.add(Embedding(vocab_size, 50, input_length=max_len))

model.add(LSTM(32))

model.add(Dropout(0.5))

model.add(Dense(16, activation="relu"))

model.add(Dropout(0.5))

model.add(Dense(3, activation="softmax"))
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
print(X_train.shape, y_train.shape, X_test.shape)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, batch_size=32, epochs=10, validation_split=0.2)
acc = model.evaluate(X_test,y_test)

print(acc)