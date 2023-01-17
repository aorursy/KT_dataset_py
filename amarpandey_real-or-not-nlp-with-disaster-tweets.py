# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import re

import nltk

import string

import numpy as np

import pandas as pd

import seaborn as sns

from tqdm import tqdm

from nltk import word_tokenize

from nltk.corpus import stopwords

from nltk.tokenize import RegexpTokenizer, TweetTokenizer

from nltk.stem import WordNetLemmatizer

from gensim.models import KeyedVectors



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
!wget 'https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz'
word_model = KeyedVectors.load_word2vec_format('/kaggle/working/GoogleNews-vectors-negative300.bin.gz', binary=True)
data = pd.read_csv("/kaggle/input/nlp-getting-started/train.csv")

test = pd.read_csv("/kaggle/input/nlp-getting-started/test.csv")



print("There are {} rows and {} columns in training data".format(data.shape[0], data.shape[1]))

print("There are {} rows and {} columns in testing data".format(test.shape[0], test.shape[1]))



data.head(3)
class_dist = data.target.value_counts()

sns.barplot(class_dist.index, class_dist)
df = pd.concat([data, test], axis=0, sort=False)

df.shape
example = "remove links from text: https://towardsdatascience.com/all- saksjak"



def remove_links(text):

    text = " ".join(list(filter(lambda l: re.match(r'https?:\/\/.*[\r\n]*', l) == None, text.split())))

    return text



remove_links(example)
df['text'] = df['text'].apply(lambda l : remove_links(l))
example = """

    I am good

    <div>This is Html</div>

    <h1>I am good</h1>

    This is blank

"""



def remove_html(text):

    regex = re.compile('<.*?>|\n')

    text = re.sub(regex, '', text)

    text = re.sub(r'[\s]+', ' ', text)

    return text



remove_html(example)
df.text = df.text.apply(lambda l : remove_html(l))
def remove_emoji(text):

    emoji_pattern = re.compile("["

                           u"\U0001F600-\U0001F64F"  # emoticons

                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs

                           u"\U0001F680-\U0001F6FF"  # transport & map symbols

                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)

                           u"\U00002702-\U000027B0"

                           u"\U000024C2-\U0001F251"

                           "]+", flags=re.UNICODE)

    return emoji_pattern.sub(r'', text)
df.text = df.text.apply(lambda l : remove_emoji(l))
tknzr = TweetTokenizer(strip_handles=True, reduce_len=True)
def remove_stopwords(text):

    text = " ".join([w.lower() for w in tknzr.tokenize(text) if w.lower() not in stopwords.words('english')])

    return text
df.text = df.text.apply(lambda l : remove_stopwords(l))
def remove_punctuation(text):

    text = ' '.join([w.lower() for w in tknzr.tokenize(text) if w.lower() not in string.punctuation])

    return text
df.text = df.text.apply(lambda l : remove_punctuation(l))
def remove_punct(text):

    table=str.maketrans('','',string.punctuation)

    return text.translate(table)
df.text = df.text.apply(lambda l : remove_punct(l))
lemmatizer = WordNetLemmatizer()



def word_lemmatizer(text):

    text = " ".join([lemmatizer.lemmatize(w) for w in tknzr.tokenize(text)])

    return text
df.text = df.text.apply(lambda l : word_lemmatizer(l))
def create_corpus(df):

    corpus = []

    max_len = -1

    for text in tqdm(df.text):

        words = [word.lower() for word in word_tokenize(text)]

        corpus.append(words)

        if max_len < len(words):

            max_len = len(words)

    

    return (max_len, corpus)
(max_len, corpus) = create_corpus(df)
from keras import optimizers

from keras.models import Sequential

from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from keras.layers import Dense, Activation, Dropout, LSTM, GRU, Bidirectional

from keras.initializers import Constant

from keras.layers.embeddings import Embedding
tokenizer = Tokenizer()

tokenizer.fit_on_texts(corpus)

sequences = tokenizer.texts_to_sequences(corpus)



padded_text = pad_sequences(sequences, maxlen=max_len, padding='post', truncating='post')

y = data.target.values
EMBEDDING_DIM = 300

word_index = tokenizer.word_index

num_words = len(word_index) + 1



emb_matrix = np.zeros((num_words, EMBEDDING_DIM))



for word, index in word_index.items():

    if index > num_words:

        continue

    try:

        emb_vector = word_model.get_vector(word)

        if emb_vector is not None:

            emb_matrix[index] = emb_vector

    except:

        continue
train = padded_text[:data.shape[0]]

test = padded_text[data.shape[0]:]



print(y.shape)

print(train.shape)

print(test.shape)
model = Sequential()



embedding_layer = Embedding(num_words,

                      EMBEDDING_DIM,

                      embeddings_initializer=Constant(emb_matrix),

                      input_length = max_len,

                      trainable = False)
model.add(embedding_layer)
model.add(Bidirectional(LSTM(512, dropout=0.5, recurrent_dropout=0.3, return_sequences=True)))

model.add(Bidirectional(LSTM(256, dropout=0.5, recurrent_dropout=0.3, return_sequences=True)))

model.add(Bidirectional(LSTM(256, dropout=0.5, recurrent_dropout=0.3, return_sequences=True)))

model.add(Bidirectional(LSTM(128, dropout=0.5, recurrent_dropout=0.3)))
model.add(Dense(512, activation='relu'))

model.add(Dropout(0.3))

model.add(Dense(512, activation='relu'))

model.add(Dropout(0.2))

model.add(Dense(512, activation='relu'))

model.add(Dropout(0.2))

model.add(Dense(1, activation='sigmoid'))
adam = optimizers.Adam(learning_rate=0.0001)

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
model.fit(train, y, batch_size=16, epochs=9, validation_split=0.1, verbose=1, shuffle=True)
prediction = model.predict(test)

prediction = np.squeeze(prediction)

prediction = np.round(prediction).astype(int).reshape(3263)
sample_sub=pd.read_csv('/kaggle/input/nlp-getting-started/sample_submission.csv')

submit = pd.DataFrame({'id':sample_sub['id'].values.tolist(),'target':prediction})

submit.to_csv('submission.csv',index=False)
x = submit.target.value_counts()

sns.barplot(x.index, x)