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
import tensorflow.keras.layers as L
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout
import nltk
from sklearn.metrics import accuracy_score
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from sklearn.metrics import confusion_matrix
import seaborn as sns
import numpy as np
from string import punctuation
from nltk.corpus import stopwords
import re
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
df = pd.read_csv("/kaggle/input/covid-19-nlp-text-classification/Corona_NLP_train.csv",encoding="latin-1")
test_data = pd.read_csv("/kaggle/input/covid-19-nlp-text-classification/Corona_NLP_test.csv")

df.head()

df['Sentiment'].value_counts()

def encode_sentiment(sentiment):
  if sentiment=="Neutral":
    return 0
  elif sentiment=="Positive":
    return 2
  elif sentiment=="Negative":
    return 1
  elif sentiment=="Extremely Positive":
    return 2
  elif sentiment=="Extremely Negative":
    return 1
df['new_sentiment'] = df['Sentiment'].apply(encode_sentiment)

df['new_sentiment'].value_counts()

train_df = df[['OriginalTweet','new_sentiment']]

train_df.head()

train_df = train_df.sample(frac=1)

sentiment = train_df['new_sentiment'].values

def process_text(text):
  text = str(text) #Convert string to str
  #Lowers the string
  text = text.lower()
  #Removes the full url
  url_remove = re.compile(r'(http|ftp|https)://([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?')
  text = re.sub(url_remove,' ',text)
  #Removes the punctuation
  text = ''.join([string for string in text if string not in punctuation and not string.isdigit()])
  #Removes any more special characters
  special_character = re.compile(r'[^a-zA-Z]')
  text = re.sub(special_character,' ', text)
  text = text.strip() #Strip white spaces
  text = text.split(' ')
  text = ' '.join([string for string in text if string not in stopwords.words('english')])#Removing all stop words
  return text
train_df['processed_tweet'] = train_df['OriginalTweet'].apply(process_text)
train_df.head()
tokenizer = Tokenizer()
tokenizer.fit_on_texts(train_df['processed_tweet'].values)
train =  tokenizer.texts_to_sequences(train_df['processed_tweet'].values)
train = pad_sequences(train,padding='post')
x_train , x_test , y_train , y_test = train_test_split(train,sentiment,test_size=0.2,random_state=42)
embedding_vectors = 30
VOCAB_SIZE = len(tokenizer.word_index)+1
model = tf.keras.Sequential([
    L.Embedding(VOCAB_SIZE,embedding_vectors, input_length=x_train.shape[1]),
    L.Bidirectional(L.LSTM(256,return_sequences=True)),
    L.GlobalMaxPool1D(),
    L.Dropout(0.4),
    L.Dense(256, activation="relu"),
    L.Dropout(0.4),
    L.Dense(3)
])
model.compile(loss=SparseCategoricalCrossentropy(from_logits=True),
              optimizer='adam',metrics=['accuracy']
             )
model.summary()
model.fit(x_train,y_train,epochs=2,
          validation_data=(x_test,y_test))
test = test_data[['OriginalTweet','Sentiment']]
test['processed_tweet'] = test_data['OriginalTweet'].apply(process_text)
test['new_sentiment'] = test_data['Sentiment'].apply(encode_sentiment)
test.drop(['OriginalTweet','Sentiment'],inplace=True,axis=1)
test.head()
test_tweet = test['processed_tweet'].values
test_sentiment = test['new_sentiment'].values
convert_seq = tokenizer.texts_to_sequences(test_tweet)
convert_seq = pad_sequences(convert_seq,padding='post')
predict = model.predict_classes(convert_seq)
print(f"The accuracy is : { accuracy_score(test_sentiment,predict)*100}%")
sns.heatmap(confusion_matrix(test_sentiment,predict),annot=True, fmt="d")