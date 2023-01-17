# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import warnings

warnings.filterwarnings('ignore')



# Modules for data manipulation

import numpy as np

import pandas as pd

import re



# Modules for visualization

import matplotlib.pyplot as plt

import seaborn as sb



# Tools for preprocessing input data

from bs4 import BeautifulSoup

from nltk import word_tokenize

from nltk.corpus import stopwords

from nltk.stem import WordNetLemmatizer



# Tools for creating ngrams and vectorizing input data

from gensim.models import Word2Vec, Phrases



# Tools for building a model

from sklearn.model_selection import train_test_split

from keras.models import Sequential

from keras.layers import Dense, LSTM, Dropout, Bidirectional

from keras.layers.embeddings import Embedding

from keras.preprocessing.sequence import pad_sequences



# Tools for assessing the quality of model prediction

from sklearn.metrics import accuracy_score, confusion_matrix





import re

from nltk.stem import WordNetLemmatizer

from nltk.corpus import stopwords



from importlib import reload

import sys

from imp import reload



if sys.version[0] == '2':

    reload(sys)

    sys.setdefaultencoding("utf-8")
SMALL_SIZE = 12

MEDIUM_SIZE = 14

BIG_SIZE = 16

LARGE_SIZE = 20



params = {

    'figure.figsize': (16, 8),

    'font.size': SMALL_SIZE,

    'xtick.labelsize': MEDIUM_SIZE,

    'ytick.labelsize': MEDIUM_SIZE,

    'legend.fontsize': BIG_SIZE,

    'figure.titlesize': LARGE_SIZE,

    'axes.titlesize': MEDIUM_SIZE,

    'axes.labelsize': BIG_SIZE

}

plt.rcParams.update(params)
#导入数据

df1 = pd.read_csv('/kaggle/input/bag-of-words-meets-bags-of-popcorn/labeledTrainData.tsv', delimiter="\t")

df1 = df1.drop(['id'], axis=1)

df1.head()
#导入数据

df2 = pd.read_csv('/kaggle/input/imdb-review-dataset/imdb_master.csv',encoding="latin-1")

df2.head()
df2 = df2.drop(['Unnamed: 0','type','file'],axis=1)

df2.columns = ["review","sentiment"]

df2.head()
df2 = df2[df2.sentiment != 'unsup']

df2['sentiment'] = df2['sentiment'].map({'pos': 1, 'neg': 0})

df2.head()
df = pd.concat([df1, df2]).reset_index(drop=True)

df.head()
df.info()


plt.hist(df[df.sentiment == 1].sentiment,

         bins=2, color='green', label='Positive')

plt.hist(df[df.sentiment == 0].sentiment,

         bins=2, color='blue', label='Negative')

plt.title('Classes distribution in the train data', fontsize=MEDIUM_SIZE)

plt.xticks([])

plt.xlim(-0.5, 2)

plt.legend()

plt.show()
stop_words = set(stopwords.words("english")) 

lemmatizer = WordNetLemmatizer()





def clean_text(text):

    text = re.sub(r'[^\w\s]','',text, re.UNICODE)

    text = text.lower()

    text = [lemmatizer.lemmatize(token) for token in text.split(" ")]

    text = [lemmatizer.lemmatize(token, "v") for token in text]

    text = [word for word in text if not word in stop_words]

    text = " ".join(text)

    return text



##



def lemmatize(tokens: list) -> list:

    # 1. Lemmatize 词形还原 去掉单词的词缀 比如，单词“cars”词形还原后的单词为“car”，单词“ate”词形还原后的单词为“eat”

    tokens = list(map(lemmatizer.lemmatize, tokens))

    lemmatized_tokens = list(map(lambda x: lemmatizer.lemmatize(x, "v"), tokens))

    # 2. Remove stop words 删除停用词

    meaningful_words = list(filter(lambda x: not x in stop_words, lemmatized_tokens))

    return meaningful_words





def preprocess(review: str, total: int, show_progress: bool = True) -> list:

    if show_progress:

        global counter

        counter += 1

        print('Processing... %6i/%6i'% (counter, total), end='\r')

    # 1. Clean text

    review = clean_review(review)

    # 2. Split into individual words

    tokens = word_tokenize(review)

    # 3. Lemmatize

    lemmas = lemmatize(tokens)

    # 4. Join the words back into one string separated by space,

    # and return the result.

    return lemmas

##

df['Processed_Reviews'] = df.review.apply(lambda x: clean_text(x))

df.head()
df.Processed_Reviews.apply(lambda x: len(x.split(" "))).mean()
from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from keras.layers import Dense , Input , LSTM , Embedding, Dropout , Activation, GRU, Flatten

from keras.layers import Bidirectional, GlobalMaxPool1D

from keras.models import Model, Sequential

from keras.layers import Convolution1D

from keras import initializers, regularizers, constraints, optimizers, layers



max_features = 6000

tokenizer = Tokenizer(num_words=max_features)

tokenizer.fit_on_texts(df['Processed_Reviews'])

list_tokenized_train = tokenizer.texts_to_sequences(df['Processed_Reviews'])



maxlen = 130

X_t = pad_sequences(list_tokenized_train, maxlen=maxlen)

y = df['sentiment']



embed_size = 128

model = Sequential()

model.add(Embedding(max_features, embed_size))

model.add(Bidirectional(LSTM(32, return_sequences = True)))

model.add(GlobalMaxPool1D())

model.add(Dense(20, activation="relu"))

model.add(Dropout(0.05))

model.add(Dense(1, activation="sigmoid"))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])



batch_size = 100

epochs = 3

model.fit(X_t,y, batch_size=batch_size, epochs=epochs, validation_split=0.2)
df_test=pd.read_csv("/kaggle/input/testdata/TestData.tsv",header=0, delimiter="\t", quoting=3)

df_test.head()

df_test["review"]=df_test.review.apply(lambda x: clean_text(x))

df_test["sentiment"] = df_test["id"].map(lambda x: 1 if int(x.strip('"').split("_")[1]) >= 5 else 0)

y_test = df_test["sentiment"]

list_sentences_test = df_test["review"]

list_tokenized_test = tokenizer.texts_to_sequences(list_sentences_test)

X_te = pad_sequences(list_tokenized_test, maxlen=maxlen)

prediction = model.predict(X_te)

y_pred = (prediction > 0.5)

from sklearn.metrics import f1_score, confusion_matrix

print('F1-score: {0}'.format(f1_score(y_pred, y_test)))

print('Confusion matrix:')

confusion_matrix(y_pred, y_test)
y_pred = model.predict(X_te)

def submit(predictions):

    df_test['sentiment'] = predictions

    df_test.to_csv('submission.csv', index=False, columns=['id','sentiment'])



submit(y_pred)