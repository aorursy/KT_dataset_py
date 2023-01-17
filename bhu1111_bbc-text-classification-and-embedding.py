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
# Importing the libraries

import numpy as np

import matplotlib.pyplot as plt

import pandas as pd



# Disable Warnings

import warnings

warnings.filterwarnings('ignore')



# NLP functionalities and libraries

import re

import nltk

import string

from nltk.corpus import stopwords

from nltk.stem import WordNetLemmatizer, SnowballStemmer

from nltk.stem.porter import PorterStemmer

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer



# Classification evaluation

from sklearn.metrics import (confusion_matrix, accuracy_score, precision_score, 

                             recall_score, f1_score)



# Word Cloud

#from wordcloud import WordCloud, STOPWORDS
# Importing the dataset

dataset = pd.read_csv('/kaggle/input/bbc-text/bbc_text.csv')
dataset.head()
dataset['category'].value_counts()
dataset.shape
dataset['text'][0]
re.sub('[^a-zA-Z]',' ', "Wow... love this place!")
news = re.sub('[^a-zA-Z]',' ', dataset['text'][0])

news
news = news.lower()

news
news = news.split()

news
len(news)
stopword_list=stopwords.words('english')

stopword_list
ss = SnowballStemmer(language='english')

news = [ss.stem(word) for word in news if not word in set(stopwords.words('english'))]

len(news)
news
news = ' '.join(news)

news
def clean_text(text):

    news = re.sub('[^a-zA-Z]', ' ',text)

    news = news.lower()

    news = news.split()

    ss = SnowballStemmer(language='english')

    news = [ss.stem(word) for word in news if not word in set(stopwords.words('english'))]

    return ' '.join(news)
dataset["clean_text"] = dataset["text"].map(lambda x: clean_text(x))

dataset.head()
corpus = dataset["clean_text"].tolist()
corpus[:5]
# Creating the Bag of Words model with Count Vectors

cv = CountVectorizer(max_features=5000)
#vocab = cv.vocabulary_

X = cv.fit_transform(corpus).toarray()

y = dataset['category'].values
vocab_cv = cv.vocabulary_
vocab_cv
len(vocab_cv)
X.shape
X[0,:]
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

y = le.fit_transform(y)

y
le.classes_
# Splitting the dataset into the Training set and Test set

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)
# Logistic Reegression

from sklearn.linear_model import LogisticRegression

lr_cv = LogisticRegression()

lr_cv.fit(X_train, y_train)
from sklearn.naive_bayes import MultinomialNB

mnb_cv = MultinomialNB()

mnb_cv.fit(X_train, y_train)
# Predicting the Test set results

y_pred_lr = lr_cv.predict(X_test)

y_pred_mnb = mnb_cv.predict(X_test)
# Making the Confusion Matrix

from sklearn.metrics import confusion_matrix,accuracy_score,classification_report,roc_auc_score,roc_curve

cm = confusion_matrix(y_test, y_pred_lr)

cm
ac = accuracy_score(y_test, y_pred_lr)

ac
print(classification_report(y_test, y_pred_lr))
cm = confusion_matrix(y_test, y_pred_mnb)

cm
ac = accuracy_score(y_test, y_pred_mnb)

ac
print(classification_report(y_test, y_pred_mnb))
# Creating the Bag of Words model Tf-Idf

tfidf = TfidfVectorizer(max_features=5000)

X = tfidf.fit_transform(corpus).toarray()

vocab_tf = tfidf.vocabulary_
len(vocab_tf)
# Splitting the dataset into the Training set and Test set

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)
# Logistic Reegression

lr_tf = LogisticRegression()

lr_tf.fit(X_train, y_train)
mnb_tf = MultinomialNB()

mnb_tf.fit(X_train, y_train)
# Predicting the Test set results

y_pred_lr = lr_tf.predict(X_test)

y_pred_mnb = mnb_tf.predict(X_test)
cm = confusion_matrix(y_test, y_pred_lr)

cm
ac = accuracy_score(y_test, y_pred_lr)

ac
print(classification_report(y_test, y_pred_lr))
cm = confusion_matrix(y_test, y_pred_mnb)

cm
ac = accuracy_score(y_test, y_pred_mnb)

ac
print(classification_report(y_test, y_pred_mnb))
import gensim

from gensim import corpora, models

from gensim.utils import simple_preprocess

from gensim.parsing.preprocessing import STOPWORDS

from nltk.stem import WordNetLemmatizer, SnowballStemmer

from nltk.stem.porter import *

import numpy as np

np.random.seed(2018)

from sklearn.cluster import KMeans

from sklearn.linear_model import LogisticRegression

# Others

import re

import nltk

import string

import numpy as np

import pandas as pd

from nltk.corpus import stopwords



from sklearn.manifold import TSNE



from nltk.stem import WordNetLemmatizer

from sklearn.linear_model import LogisticRegression

from bs4 import BeautifulSoup as soup

from nltk.stem.snowball import SnowballStemmer
# Keras

from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from keras.models import Sequential, load_model, Model

from keras.utils import plot_model

from keras.layers import Flatten, Dropout, Activation, Input, Dense, concatenate

from keras.layers.embeddings import Embedding

from keras.initializers import Constant
dummy_y = pd.get_dummies(dataset['category']).values

dummy_y[:10]
dummy_y.shape
def clean_text(text):

    

    ## Remove puncuation

    text = text.translate(string.punctuation)

    

    ## Convert words to lower case and split them

    text = text.lower().split()

    

    ## Remove stop words

    stops = set(stopwords.words("english"))

    text = [w for w in text if not w in stops and len(w) > 3]

    

    text = " ".join(text)

    

    ## Clean the text

    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)

    text = re.sub(r"what's", "what is ", text)

    text = re.sub(r"\'s", " ", text)

    text = re.sub(r"\'ve", " have ", text)

    text = re.sub(r"n't", " not ", text)

    text = re.sub(r"i'm", "i am ", text)

    text = re.sub(r"\'re", " are ", text)

    text = re.sub(r"\'d", " would ", text)

    text = re.sub(r"\'ll", " will ", text)

    text = re.sub(r",", " ", text)

    text = re.sub(r"\.", " ", text)

    text = re.sub(r"!", " ! ", text)

    text = re.sub(r"\/", " ", text)

    text = re.sub(r"\^", " ^ ", text)

    text = re.sub(r"\+", " + ", text)

    text = re.sub(r"\-", " - ", text)

    text = re.sub(r"\=", " = ", text)

    text = re.sub(r"'", " ", text)

    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)

    text = re.sub(r":", " : ", text)

    text = re.sub(r" e g ", " eg ", text)

    text = re.sub(r" b g ", " bg ", text)

    text = re.sub(r" u s ", " american ", text)

    text = re.sub(r"\0s", "0", text)

    text = re.sub(r" 9 11 ", "911", text)

    text = re.sub(r"e - mail", "email", text)

    text = re.sub(r"j k", "jk", text)

    text = re.sub(r"\s{2,}", " ", text)

    

    ## Stemming

    text = text.split()

    stemmer = SnowballStemmer('english')

    stemmed_words = [stemmer.stem(word) for word in text]

    text = " ".join(stemmed_words)

    return text
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(dataset['clean_text'].values, 

                                                    dummy_y, 

                                                    test_size = 0.20, 

                                                    random_state = 0)
tokenizer = Tokenizer()

tokenizer.fit_on_texts(X_train)
tokenizer.word_index
trainsequences = tokenizer.texts_to_sequences(X_train)

print(trainsequences)
X_train.shape
len(trainsequences)
avg_len = [len(seq) for seq in trainsequences]

mean_len = np.mean(avg_len)

mean_len
MAXLEN = 220
trainseqs = pad_sequences(trainsequences, maxlen=MAXLEN, padding='post')

print(trainseqs)
trainseqs.shape
testsequences = tokenizer.texts_to_sequences(X_test)

testseqs = pad_sequences(testsequences, maxlen=MAXLEN, padding='post')
print(testseqs)
testseqs.shape
y_test.shape
EMBEDDING_SIZE = 8
VOCAB_SIZE = len(tokenizer.word_index) + 1

VOCAB_SIZE
OP_UNITS = dataset['category'].nunique()
# define the model

model = Sequential()

embedding_layer = Embedding(VOCAB_SIZE,

                            EMBEDDING_SIZE,

                            input_length=MAXLEN)

model.add(embedding_layer)

model.add(Flatten())

model.add(Dense(256, activation='relu'))

model.add(Dense(128, activation='relu'))

model.add(Dense(64, activation='relu'))

model.add(Dense(OP_UNITS, activation='softmax'))



# compile the model

model.compile(optimizer='adam', 

              loss='categorical_crossentropy', 

              metrics=['accuracy'])
print(model.summary())
# fit the model

history = model.fit(trainseqs,

                    y_train,

                    epochs=20,

                    batch_size=64,

                    validation_data=(testseqs,y_test),

                    verbose=1).history
res_df = pd.DataFrame(history)

res_df.head()
# Plot training vs validation Loss

plt.plot(res_df['loss'],label="Training")

plt.plot(res_df['val_loss'],label="Validation")

plt.legend(loc='best')

plt.xlabel('Epochs')

plt.ylabel('Loss')

plt.title('Training vs Validation Loss')
# Plot training vs validation Accuracy

plt.plot(res_df['accuracy'],label="Training")

plt.plot(res_df['val_accuracy'],label="Validation")

plt.legend(loc='best')

plt.xlabel('Epochs')

plt.ylabel('Accuracy')

plt.title('Training vs Validation Accuracy')
model.layers
model.layers[0].get_weights()[0].shape
model.layers[0].get_weights()[0][0]
# Extract weights from the Embedding Layers

embeddings = model.layers[0].get_weights()[0]



# `embeddings` has a shape of (num_vocab, embedding_dim) 



# `word_to_index` is a mapping (i.e. dict) from words to 

# their index

words_embeddings = {w:embeddings[idx - 1] for w, idx in tokenizer.word_index.items()}
words_embeddings['play']