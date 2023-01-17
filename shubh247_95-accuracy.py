import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import plotly.offline as py

import warnings

warnings.filterwarnings('ignore')

import os

os.listdir("../input")
data = pd.read_csv('../input/GrammarandProductReviews.csv')
data.head()
data.shape
data.info()
data.isnull().sum()
data = data.dropna(subset = ['reviews.text'])
from wordcloud import WordCloud, STOPWORDS

stopwords = set(STOPWORDS)



def show_wordcloud(data, title=None):

    wordcloud = WordCloud(

    background_color = 'black',

    stopwords=stopwords,

    max_words=200,

    max_font_size=40,

    scale=3,

    random_state=1 #choose at random

    ).generate(str(data))

    

    fig = plt.figure(1, figsize=(15,15))

    plt.axis('off')

    if title:

        fig.suptitle(title, fontsize=20)

        fig.subplots_adjust(top = 2.3)

        

    plt.imshow(wordcloud)

    plt.show()

    

show_wordcloud(data['reviews.text'])

        
data['reviews.rating'].value_counts()

data['reviews_length'] = data['reviews.text'].apply(len)
sns.set(font_scale = 2.0)



g = sns.FacetGrid(data, col='reviews.rating', size=5)

g.map(plt.hist, 'reviews_length')
data['reviews.didPurchase'].fillna('Review N/A', inplace=True)
plt.figure(figsize=(10,8))

ax = sns.countplot(data['reviews.didPurchase'])

ax.set_xlabel(xlabel='Peoples Reviews', fontsize=17)

ax.set_ylabel(ylabel='No. of Reviews', fontsize=17)

ax.axes.set_title('Genuine No. of Reviews', fontsize=17)

ax.tick_params(labelsize=13)
sns.set(font_scale=1.4)

plt.figure(figsize = (10,5))

sns.heatmap(data.corr(), cmap='coolwarm', annot=True, linewidth=.5)
from sklearn.model_selection import cross_val_score

from scipy.sparse import hstack

from sklearn.feature_extraction.text import TfidfVectorizer
all_text = data['reviews.text']

train_text = data['reviews.text']

y=data['reviews.rating']
word_vectorizer = TfidfVectorizer(

    sublinear_tf=True,

    strip_accents='unicode',

    analyzer='word',

    token_pattern=r'\w{1,}',

    stop_words='english',

    ngram_range=(1,1),

    max_features=10000)

word_vectorizer.fit(all_text)

train_word_features = word_vectorizer.transform(train_text)

char_vectorizer = TfidfVectorizer(

    sublinear_tf= True,

    strip_accents='unicode',

    analyzer='char',

    stop_words='english',

    ngram_range=(2,6),

    max_features=50000)

char_vectorizer.fit(all_text)

train_char_features = char_vectorizer.transform(train_text)



train_features = hstack([train_char_features, train_word_features])
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(

train_features, y, test_size=0.3, random_state=101)
from sklearn.preprocessing import LabelEncoder

# if any categorical data has available, convert to numerical

Encoder = LabelEncoder()

y_train = Encoder.fit_transform(y_train)

y_test = Encoder.fit_transform(y_test)

from sklearn.ensemble import RandomForestClassifier



classifier = RandomForestClassifier()

classifier.fit(X_train, y_train)

r_pred = classifier.predict(X_test)
from sklearn.metrics import accuracy_score

rf_accuracy = accuracy_score(r_pred, y_test)
print('random forest model accuracy is', rf_accuracy)
from sklearn.svm import LinearSVC
model = LinearSVC()
model.fit(X_train, y_train)
s_pred = model.predict(X_test)
svm_accuracy = accuracy_score(s_pred, y_test)
print('svm accuracy is', svm_accuracy)
data['sentiment'] = data['reviews.rating']<4
from sklearn.model_selection import train_test_split

train_text, test_text, train_y, test_y = train_test_split(

    data['reviews.text'], data['sentiment'], test_size=0.2)
from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from keras.models import Sequential

from keras.layers import Dense, Embedding, LSTM

from keras.utils.np_utils import to_categorical

from keras.callbacks import ModelCheckpoint

from keras.models import load_model 

from keras.optimizers import Adam
MAX_NB_WORDS = 20000



#get the text

texts_train = train_text.astype(str)

texts_test = test_text.astype(str)



#vectorize sample into 2d int

tokenizer = Tokenizer(nb_words = MAX_NB_WORDS,

                     char_level=False)

tokenizer.fit_on_texts(texts_train)

sequences = tokenizer.texts_to_sequences(texts_train)

sequences_test = tokenizer.texts_to_sequences(texts_test)



word_index = tokenizer.word_index

print('Found %s unique tokens.' % len(word_index))
MAX_SEQUENCE_LENGTH = 200

#pad sequences are used to bring all sentences to same size

#pad sequence with o's

x_train = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

x_test = pad_sequences(sequences_test, maxlen=

                      MAX_SEQUENCE_LENGTH)

print('shape of data tensor:', x_train.shape)

print('shape of data test tensor:', x_test.shape)

model = Sequential()

model.add(Embedding(MAX_NB_WORDS, 128))

model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2, input_shape =(1,)))

model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x_train, train_y,batch_size=64, epochs = 20, validation_data=(x_test, test_y))