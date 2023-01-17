# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import gensim

import os

import re

from tensorflow.python.keras.preprocessing.text import Tokenizer

from tensorflow.python.keras.preprocessing.sequence import pad_sequences

from keras.models import Sequential

from keras.layers import Dense, Embedding, GRU, LSTM

from keras.layers.embeddings import Embedding

from keras.initializers import Constant

from keras.callbacks import ModelCheckpoint

from keras.models import load_model

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/nlp-getting-started/train.csv")

data.head()
def clean_text(text):

    text = text.lower()

    

    pattern = re.compile('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')

    text = pattern.sub('', text)

    text = " ".join(filter(lambda x:x[0]!='@', text.split()))

    emoji = re.compile("["

                           u"\U0001F600-\U0001FFFF"  # emoticons

                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs

                           u"\U0001F680-\U0001F6FF"  # transport & map symbols

                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)

                           u"\U00002702-\U000027B0"

                           u"\U000024C2-\U0001F251"

                           "]+", flags=re.UNICODE)

    

    text = emoji.sub(r'', text)

    text = text.lower()

    text = re.sub(r"i'm", "i am", text)

    text = re.sub(r"he's", "he is", text)

    text = re.sub(r"she's", "she is", text)

    text = re.sub(r"that's", "that is", text)        

    text = re.sub(r"what's", "what is", text)

    text = re.sub(r"where's", "where is", text) 

    text = re.sub(r"\'ll", " will", text)  

    text = re.sub(r"\'ve", " have", text)  

    text = re.sub(r"\'re", " are", text)

    text = re.sub(r"\'d", " would", text)

    text = re.sub(r"\'ve", " have", text)

    text = re.sub(r"won't", "will not", text)

    text = re.sub(r"don't", "do not", text)

    text = re.sub(r"did't", "did not", text)

    text = re.sub(r"can't", "can not", text)

    text = re.sub(r"it's", "it is", text)

    text = re.sub(r"couldn't", "could not", text)

    text = re.sub(r"have't", "have not", text)

    text = re.sub(r"[,.\"\'!@#$%^&*(){}?/;`~:<>+=-]", "", text)

    return text
import string

from nltk.tokenize import word_tokenize

from nltk.corpus import stopwords



def CleanTokenize(df):

    tweet_lines = list()

    lines = df["text"].values.tolist()



    for line in lines:

        line = clean_text(line)

        # tokenize the text

        tokens = word_tokenize(line)

    #     tokens = [w.lower() for w in tokens]

        # remove puntuations

        table = str.maketrans('', '', string.punctuation)

        stripped = [w.translate(table) for w in tokens]

        # remove non alphabetic characters

        words = [word for word in stripped if word.isalpha()]

        stop_words = set(stopwords.words("english"))

        words = [w for w in words if not w in stop_words]

        tweet_lines.append(words)

    return tweet_lines



tweet_lines = CleanTokenize(data)

tweet_lines[0:10]
# import string

# fake_lines = ["#RockyFire Update => California Hwy. 20 closed in both directions due to Lake County fire - #CAfire #wildfires",

#               "@bbcmtd Wholesale Markets ablaze https://t.co/lHYXEOHY6C"]



# fake_tweet_lines = list()

# for line in fake_lines:

#     line = clean_text(str(line))

#     # tokenize the text

#     tokens = word_tokenize(line)

#     #  tokens = [w.lower() for w in tokens]

#     # remove puntuations

#     table = str.maketrans('', '', string.punctuation)

#     stripped = [w.translate(table) for w in tokens]

#     # remove non alphabetic characters

#     words = [word for word in stripped if word.isalpha()]

#     stop_words = set(stopwords.words("english"))

#     words = [w for w in words if not w in stop_words]

#     fake_tweet_lines.append(words)

    

# fake_tweet_lines
from collections import Counter

from wordcloud import WordCloud, ImageColorGenerator

import matplotlib.pyplot as plt

all_tweets = [j for sub in tweet_lines for j in sub] 

word_could_dict=Counter(all_tweets)

word_could_dict.most_common(10)



wordcloud = WordCloud(width = 1000, height = 500).generate_from_frequencies(word_could_dict)

plt.figure(figsize=(15,8))

plt.imshow(wordcloud)

plt.axis("off")
pos_data = data.loc[data['target'] == 1]

pos_tweet_lines = CleanTokenize(pos_data)

pos_tweets = [j for sub in pos_tweet_lines for j in sub] 

word_could_dict=Counter(pos_tweets)

word_could_dict.most_common(10)



wordcloud = WordCloud(width = 1000, height = 500).generate_from_frequencies(word_could_dict)

plt.figure(figsize=(15,8))

plt.imshow(wordcloud)

plt.axis("off")
neg_data = data.loc[data['target'] == 0]

neg_tweet_lines = CleanTokenize(neg_data)

neg_tweets = [j for sub in neg_tweet_lines for j in sub] 

word_could_dict=Counter(neg_tweets)

word_could_dict.most_common(10)



wordcloud = WordCloud(width = 1000, height = 500).generate_from_frequencies(word_could_dict)

plt.figure(figsize=(15,8))

plt.imshow(wordcloud)

plt.axis("off")
VALIDATION_SPLIT = 0.2

max_length = 10

EMBEDDING_DIM = 120



tokenizer_obj = Tokenizer()

tokenizer_obj.fit_on_texts(tweet_lines)

sequences = tokenizer_obj.texts_to_sequences(tweet_lines)



word_index = tokenizer_obj.word_index

print('Found %s unique tokens.' % len(word_index))

vocab_size = len(tokenizer_obj.word_index) + 1

print('vocab_size '+str(vocab_size))



review_pad = pad_sequences(sequences, maxlen=max_length, padding='post')

sentiment =  data['target'].values



indices = np.arange(review_pad.shape[0])

np.random.shuffle(indices)

review_pad = review_pad[indices]

sentiment = sentiment[indices]



num_validation_samples = int(VALIDATION_SPLIT * review_pad.shape[0])



X_train_pad = review_pad[:-num_validation_samples]

y_train = sentiment[:-num_validation_samples]

X_test_pad = review_pad[-num_validation_samples:]

y_test = sentiment[-num_validation_samples:]
print('Shape of X_train_pad tensor:', X_train_pad.shape)

print('Shape of y_train tensor:', y_train.shape)



print('Shape of X_test_pad tensor:', X_test_pad.shape)

print('Shape of y_test tensor:', y_test.shape)
model = Sequential()

model.add(Embedding(vocab_size, EMBEDDING_DIM, input_length=max_length))

model.add(LSTM(units=32,  dropout=0.4, recurrent_dropout=0.3))

# model.add(Dense(64, activation='relu'))

model.add(Dense(1, activation='sigmoid'))



model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])



print('Summary of the built model...')

print(model.summary())
model.fit(X_train_pad, y_train, batch_size=32, epochs=2, validation_data=(X_test_pad, y_test), verbose=2)
test = pd.read_csv("../input/nlp-getting-started/test.csv")

len(test)
test_lines = CleanTokenize(test)

len(test_lines)

test_sequences = tokenizer_obj.texts_to_sequences(test_lines)

test_review_pad = pad_sequences(test_sequences, maxlen=max_length, padding='post')

test_review_pad.shape
predictions = model.predict(test_review_pad)
submission = pd.read_csv("../input/nlp-getting-started/sample_submission.csv")

submission["target"] = predictions

submission["target"] = submission["target"].apply(lambda x : 0 if x<=.5 else 1)
submission.to_csv("submit_2.csv", index=False)