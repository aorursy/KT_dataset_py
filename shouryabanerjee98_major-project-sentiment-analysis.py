

import numpy as np 

import pandas as pd 





import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.

%%time

train = pd.read_csv('/kaggle/input/twitter-sentiment-analysis-hatred-speech/train.csv')

test = pd.read_csv('/kaggle/input/twitter-sentiment-analysis-hatred-speech/test.csv')
train.head()
test.head()
def num_of_words(df):

    df['word_count'] = df['tweet'].apply(lambda x : len(str(x).split(" ")))

    print(df[['tweet','word_count']].head())
num_of_words(train)
num_of_words(test)
def num_of_chars(df):

    df['char_count'] = df['tweet'].str.len() ## this also includes spaces

    print(df[['tweet','char_count']].head())
num_of_chars(train)
num_of_chars(test)
def avg_word(sentence):

    words = sentence.split()    

    return (sum(len(word) for word in words)/len(words))
def avg_word_length(df):

    df['avg_word'] = df['tweet'].apply(lambda x: avg_word(x))

    print(df[['tweet','avg_word']].head())
avg_word_length(train)
avg_word_length(test)
import nltk

from nltk.corpus import stopwords

set(stopwords.words('english'))
from nltk.corpus import stopwords

stop = stopwords.words('english')
def stop_words(df):

    df['stopwords'] = df['tweet'].apply(lambda x: len([x for x in x.split() if x in stop]))

    print(df[['tweet','stopwords']].head())
stop_words(train)
stop_words(test)
def hash_tags(df):

    df['hashtags'] = df['tweet'].apply(lambda x: len([x for x in x.split() if x.startswith('#')]))

    print(df[['tweet','hashtags']].head())
hash_tags(train)
hash_tags(test)
def num_numerics(df):

    df['numerics'] = df['tweet'].apply(lambda x: len([x for x in x.split() if x.isdigit()]))

    print(df[['tweet','numerics']].head())
num_numerics(train)
num_numerics(test)
def num_uppercase(df):

    df['upper_case'] = df['tweet'].apply(lambda x: len([x for x in x.split() if x.isupper()]))

    print(df[['tweet','upper_case']].head())
num_uppercase(train)
num_uppercase(test)
from sklearn.feature_extraction.text import CountVectorizer

corpus = [

          'This is the first document.',

          'This document is the second document.',

          'And this is the third one.',

          'Is this the first document?',

         ]



vectorizer = CountVectorizer()

X = vectorizer.fit_transform(corpus)

print(vectorizer.get_feature_names())
print(X.toarray())
vectorizer2 = CountVectorizer(analyzer='word', ngram_range=(2, 2))

X2 = vectorizer2.fit_transform(corpus)

print(vectorizer2.get_feature_names())
print(X2.toarray())
from sklearn.feature_extraction.text import HashingVectorizer

corpus = [

          'This is the first document.',

          'This document is the second document.',

          'And this is the third one.',

          'Is this the first document?',

         ]

vectorizer = HashingVectorizer(n_features=2**4)

X = vectorizer.fit_transform(corpus)

print(X.shape)
def lower_case(df):

    df['tweet'] = df['tweet'].apply(lambda x: " ".join(x.lower() for x in x.split()))

    print(df['tweet'].head())
lower_case(train)
lower_case(test)
def punctuation_removal(df):

    df['tweet'] = df['tweet'].str.replace('[^\w\s]','')

    print(df['tweet'].head())
punctuation_removal(train)
punctuation_removal(test)
from nltk.corpus import stopwords

stop = stopwords.words('english')
def stop_words_removal(df):

    df['tweet'] = df['tweet'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))

    print(df['tweet'].head())
stop_words_removal(train)
stop_words_removal(test)
freq = pd.Series(' '.join(train['tweet']).split()).value_counts()[:10]

freq
freq = list(freq.index)
def frequent_words_removal(df):    

    df['tweet'] = df['tweet'].apply(lambda x: " ".join(x for x in x.split() if x not in freq))

    print(df['tweet'].head())
frequent_words_removal(train)
frequent_words_removal(test)
freq = pd.Series(' '.join(train['tweet']).split()).value_counts()[-10:]

freq
freq = list(freq.index)
def rare_words_removal(df):

    df['tweet'] = df['tweet'].apply(lambda x: " ".join(x for x in x.split() if x not in freq))

    print(df['tweet'].head())
rare_words_removal(train)
rare_words_removal(test)
from textblob import TextBlob
def spell_correction(df):

    return df['tweet'][:5].apply(lambda x: str(TextBlob(x).correct()))
spell_correction(train)
spell_correction(test)
def tokens(df):

    return TextBlob(df['tweet'][1]).words
tokens(train)
tokens(test)
from nltk.stem import PorterStemmer

st = PorterStemmer()
def stemming(df):

    return df['tweet'][:5].apply(lambda x: " ".join([st.stem(word) for word in x.split()]))
stemming(train)
stemming(test)
from textblob import Word
def lemmatization(df):

    df['tweet'] = df['tweet'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))

    print(df['tweet'].head())
lemmatization(train)
lemmatization(test)
from textblob import TextBlob
def combination_of_words(df):

    return (TextBlob(df['tweet'][0]).ngrams(2))
combination_of_words(train)
combination_of_words(test)
def term_frequency(df):

    tf1 = (df['tweet'][1:2]).apply(lambda x: pd.value_counts(x.split(" "))).sum(axis = 0).reset_index()

    tf1.columns = ['words','tf']

    return tf1.head()
term_frequency(train)
term_frequency(test)
tf1 = (train['tweet'][1:2]).apply(lambda x: pd.value_counts(x.split(" "))).sum(axis = 0).reset_index()

tf1.columns = ['words','tf']

tf1.head()
tf2 = (test['tweet'][1:2]).apply(lambda x: pd.value_counts(x.split(" "))).sum(axis = 0).reset_index()

tf2.columns = ['words','tf']

tf2.head()
tf1 = (train['tweet'][1:2]).apply(lambda x: pd.value_counts(x.split(" "))).sum(axis = 0).reset_index()

tf1.columns = ['words','tf']
for i,word in enumerate(tf1['words']):

    tf1.loc[i, 'idf'] = np.log(train.shape[0]/(len(train[train['tweet'].str.contains(word)])))
tf1['tfidf'] = tf1['tf'] * tf1['idf']

tf1
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(max_features=1000, lowercase=True, analyzer='word',

 stop_words= 'english',ngram_range=(1,1))

train_vect = tfidf.fit_transform(train['tweet'])

train_vect
from sklearn.feature_extraction.text import CountVectorizer

bow = CountVectorizer(max_features=1000, lowercase=True, ngram_range=(1,1),analyzer = "word")

train_bow = bow.fit_transform(train['tweet'])

train_bow
def polarity_subjectivity(df):

    return df['tweet'][:5].apply(lambda x: TextBlob(x).sentiment)
polarity_subjectivity(train)
def sentiment_analysis(df):

    df['sentiment'] = df['tweet'].apply(lambda x: TextBlob(x).sentiment[0] )

    return df[['tweet','sentiment']].head()
sentiment_analysis(train)
from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

max_fatures = 2000

tokenizer = Tokenizer(num_words=max_fatures, split=' ')

tokenizer.fit_on_texts(train_model['tweet'].values)

X = tokenizer.texts_to_sequences(train_model['tweet'].values)

X = pad_sequences(X)
sentiment_analysis(test)
polarity_subjectivity(test)


train_model = train[['tweet','sentiment']]

train_model.head()


X_train, X_test, Y_train, Y_test = train_test_split(X,train[['sentiment']], test_size = 0.33, random_state = 42)

print(X_train.shape,Y_train.shape)

print(X_test.shape,Y_test.shape)
from keras.models import Sequential

from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D

from sklearn.model_selection import train_test_split

from keras.utils.np_utils import to_categorical

import re
from keras.preprocessing.sequence import pad_sequences

max_fatures = 2000

tokenizer = Tokenizer(num_words=max_fatures, split=' ')

tokenizer.fit_on_texts(train_model['tweet'].values)

X = tokenizer.texts_to_sequences(train_model['tweet'].values)

X = pad_sequences(X)
lstm_out = 196



model = Sequential()

model.add(Embedding(max_fatures, 128,input_length = 17))

model.add(SpatialDropout1D(0.4))

model.add(LSTM(lstm_out, dropout=0.2, recurrent_dropout=0.2))

model.add(Dense(2,activation='softmax'))

model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])

print(model.summary())
X_train, X_test, Y_train, Y_test = train_test_split(X,train[['sentiment']], test_size = 0.33, random_state = 42)

print(X_train.shape,Y_train.shape)

print(X_test.shape,Y_test.shape)
from keras.utils import to_categorical

train_labels = to_categorical(Y_train)

batch_size = 32

model.fit(X_train, train_labels, epochs = 7, batch_size=batch_size, verbose = 2)
validation_size = 32

X_validate = X_test[-validation_size:]

Y_validate = Y_test[-validation_size:]

X_test = X_test[:-validation_size]

Y_test = Y_test[:-validation_size]

test_labels = to_categorical( Y_test)

score,acc = model.evaluate(X_test, test_labels, verbose = 2, batch_size = batch_size)

print("score: %.2f" % (score))

print("acc: %.2f" % (acc))
data = {'No' : 1, 'tweet' : "this project is horrible"}

testline = pd.DataFrame(data, index=[0])

sentiment_analysis(testline)