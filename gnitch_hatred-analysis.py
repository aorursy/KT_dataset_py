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
import numpy as np

import pandas as pd

import nltk

import string 

import spacy

import re

from collections import Counter

from sklearn.model_selection import train_test_split

import tensorflow as tf

import tensorflow_hub as hub

from nltk.stem.porter import PorterStemmer

from nltk.corpus import stopwords

from nltk.corpus import wordnet

from nltk.stem import WordNetLemmatizer
df_train = pd.read_csv('/kaggle/input/twitter-sentiment-analysis-hatred-speech/train.csv')

df_test = pd.read_csv('/kaggle/input/twitter-sentiment-analysis-hatred-speech/test.csv')

df_train.info()
# convert to lower case

df_train['cleaned_tweet'] = df_train['tweet'].str.lower()

df_test['cleaned_tweet'] = df_test['tweet'].str.lower()

df_train['cleaned_tweet'].head(10)
# remove punctuation

df_train['cleaned_tweet'] = df_train['cleaned_tweet'].apply(lambda text : text.translate(str.maketrans('','',string.punctuation)))

df_test['cleaned_tweet'] = df_test['cleaned_tweet'].apply(lambda text : text.translate(str.maketrans('','',string.punctuation)))



# remove urls

url_pattern = re.compile(r'https?://\S+|www\.\S+')

df_train['cleaned_tweet'] = df_train['cleaned_tweet'].apply(lambda text : url_pattern.sub(r'', text))

df_test['cleaned_tweet'] = df_test['cleaned_tweet'].apply(lambda text : url_pattern.sub(r'', text))



df_train['cleaned_tweet'].head(10)
# remove stopwords

STOPWORDS = set(stopwords.words('english'))

df_train['cleaned_tweet'] = df_train['cleaned_tweet'].apply(lambda text : ' '.join([word for word in str(text).split() if word not in STOPWORDS]))

df_test['cleaned_tweet'] = df_test['cleaned_tweet'].apply(lambda text : ' '.join([word for word in str(text).split() if word not in STOPWORDS]))

df_train['cleaned_tweet'].head(10)
# remove frequent words

cnt = Counter()

cnt_test = Counter()

for text in df_train["cleaned_tweet"].values:

    for word in text.split():

        cnt[word] += 1



for text in df_test["cleaned_tweet"].values:

    for word in text.split():

        cnt_test[word] += 1        



temp_test = cnt_test.most_common(20).copy()

temp_test = [x[0] for x in temp_test ]

FREQWORDS_TEST = set(temp_test.copy())        

df_test['cleaned_tweet'] = df_test['cleaned_tweet'].apply(lambda text : ' '.join([word for word in str(text).split() if word not in FREQWORDS_TEST]))        

    

print(cnt.most_common(20))        

temp = cnt.most_common(20).copy()

temp = [x[0] for x in temp ]

FREQWORDS = set(temp.copy())

df_train['cleaned_tweet'] = df_train['cleaned_tweet'].apply(lambda text : ' '.join([word for word in str(text).split() if word not in FREQWORDS]))

df_train['cleaned_tweet'].head(10)
# remove rare words

n_rare_words = 10

RAREWORDS = set([w for (w, wc) in cnt.most_common()[:-n_rare_words-1:-1]])

RAREWORDS_TEST = set([w for (w, wc) in cnt_test.most_common()[:-n_rare_words-1:-1]])

df_test['cleaned_tweet'] = df_test['cleaned_tweet'].apply(lambda text :  ' '.join([word for word in str(text).split() if word not in RAREWORDS_TEST]))



print(RAREWORDS)

df_train['cleaned_tweet'] = df_train['cleaned_tweet'].apply(lambda text :  ' '.join([word for word in str(text).split() if word not in RAREWORDS]))

df_train['cleaned_tweet'].head(10)
# lemmatization

lemmatizer = WordNetLemmatizer()

wordnet_map = {"N":wordnet.NOUN, "V":wordnet.VERB, "J":wordnet.ADJ, "R":wordnet.ADV}

df_train["cleaned_tweet"] = df_train["cleaned_tweet"].apply(lambda text: " ".join([lemmatizer.lemmatize(word, wordnet_map.get(pos[0], wordnet.NOUN)) for word, pos in nltk.pos_tag(text.split())]))

df_test["cleaned_tweet"] = df_test["cleaned_tweet"].apply(lambda text: " ".join([lemmatizer.lemmatize(word, wordnet_map.get(pos[0], wordnet.NOUN)) for word, pos in nltk.pos_tag(text.split())]))

df_train['cleaned_tweet'].head(10)
# keep only numbers and text

df_train["cleaned_tweet"] = df_train["cleaned_tweet"].apply(lambda text: ' '.join(re.sub(r'[^a-zA-Z0-9]','',word) for word in text.split()))

df_test["cleaned_tweet"] = df_test["cleaned_tweet"].apply(lambda text: ' '.join(re.sub(r'[^a-zA-Z0-9]','',word) for word in text.split()))

df_train['cleaned_tweet'].head(10)
from sklearn.model_selection import train_test_split



X_train,X_test,y_train,y_test = train_test_split(df_train.iloc[:,3].values,df_train.iloc[:,1].values,test_size=0.25)
embedding = "https://tfhub.dev/google/nnlm-en-dim50/2"

hub_layer = hub.KerasLayer(embedding,input_shape=[],output_shape=[None,None,50], dtype=tf.string, trainable=True)
model = tf.keras.Sequential()

model.add(hub_layer)

model.add(tf.keras.layers.Dense(18, activation='relu'))

model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

model.fit(X_train, y_train,epochs = 5,steps_per_epoch=50)
model.summary()
results = model.evaluate(X_test,y_test, verbose=2)

for name, value in zip(model.metrics_names, results):

  print("%s: %.3f" % (name, value))
ans = model.predict(df_test["cleaned_tweet"])

test_labels = model.predict_classes(df_test["cleaned_tweet"])

df_test['predicted_label'] = test_labels.copy()

df_test['predicted_label'].value_counts()