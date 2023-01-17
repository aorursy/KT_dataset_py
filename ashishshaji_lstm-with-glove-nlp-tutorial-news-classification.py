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
import seaborn as sns
import matplotlib.pyplot as plt
""
%matplotlib inline
sns.set_style("darkgrid")
true = pd.read_csv("../input/fake-and-real-news-dataset/True.csv")
false = pd.read_csv("../input/fake-and-real-news-dataset/Fake.csv")
from wordcloud import WordCloud, ImageColorGenerator
text = " ".join(str(each) for each in true.text.unique())
wordcloud = WordCloud(max_words=200,colormap='Set3', background_color="black").generate(text)
plt.figure(figsize=(15,10))
plt.imshow(wordcloud, interpolation='Bilinear')
plt.axis("off")
plt.figure(1,figsize=(12, 12))
plt.show()
from wordcloud import WordCloud, ImageColorGenerator
text = " ".join(str(each) for each in false.text.unique())
wordcloud = WordCloud(max_words=200,colormap='Set3', background_color="black").generate(text)
plt.figure(figsize=(15,10))
plt.imshow(wordcloud, interpolation='Bilinear')
plt.axis("off")
plt.figure(1,figsize=(12, 12))
plt.show()
true['label'] = 1
false['label'] = 0
news = pd.concat([true,false]) 
news
news['text'] = news['text'] + " " + news['title']
news
df=news.drop(["date","title","subject"],axis=1)
df
print(false.shape)
print(true.shape)

sns.countplot(x="label", data=news);
plt.show()
import nltk
import string
from nltk.corpus import stopwords
import re
def rem_punctuation(text):
  return text.translate(str.maketrans('','',string.punctuation))

def rem_numbers(text):
  return re.sub('[0-9]+','',text)


def rem_urls(text):
  return re.sub('https?:\S+','',text)


def rem_tags(text):
  return re.sub('<.*?>'," ",text)

df['text'].apply(rem_urls)
df['text'].apply(rem_punctuation)
df['text'].apply(rem_tags)
df['text'].apply(rem_numbers)

stop = set(stopwords.words('english'))

def rem_stopwords(df_news):
    
    words = [ch for ch in df_news if ch not in stop]
    words= "".join(words).split()
    words= [words.lower() for words in df_news.split()]
    
    return words    
df['text'].apply(rem_stopwords)

from nltk.stem import WordNetLemmatizer
#nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()

def lemmatize_words(text):
  lemmas = []
  for word in text.split():
    lemmas.append(lemmatizer.lemmatize(word))
  return " ".join(lemmas)
df['text'].apply(lemmatize_words)

from keras.preprocessing.text import Tokenizer

from keras.utils import to_categorical

x = df['text'].values
y= df['label'].values
tokenizer = Tokenizer()
tokenizer.fit_on_texts(x)
word_to_index = tokenizer.word_index
x = tokenizer.texts_to_sequences(x)
vocab_size =  len(word_to_index)
oov_tok = "<OOV>"
max_length = 250
embedding_dim = 100
from keras.preprocessing.sequence import pad_sequences

x = pad_sequences(x, maxlen=max_length)
embeddings_index = {};
with open('../input/glove6b100dtxt/glove.6B.100d.txt') as f:
    for line in f:
        values = line.split();
        word = values[0];
        coefs = np.asarray(values[1:], dtype='float32');
        embeddings_index[word] = coefs;

embeddings_matrix = np.zeros((vocab_size+1, embedding_dim));
for word, i in word_to_index.items():
    embedding_vector = embeddings_index.get(word);
    if embedding_vector is not None:
        embeddings_matrix[i] = embedding_vector;
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(x,y,test_size=0.20,random_state=1)

import tensorflow as tf

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size+1, embedding_dim, input_length=max_length, weights=[embeddings_matrix], trainable=False),
    tf.keras.layers.LSTM(64,return_sequences=True),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.LSTM(32),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(24, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')

   
])
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model.summary()
epochs = 10
history = model.fit(X_train,y_train,epochs=epochs,validation_data=(X_test,y_test),batch_size=128)
result = model.evaluate(X_test, y_test)
# extract those
loss = result[0]
accuracy = result[1]


print(f"[+] Accuracy: {accuracy*100:.2f}%")