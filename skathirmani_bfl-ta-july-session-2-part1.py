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
imdb = pd.read_csv('https://raw.githubusercontent.com/skathirmani/datasets/master/imdb_sentiment.csv')
imdb.head()
from gensim.parsing.preprocessing import remove_stopwords
from gensim.parsing.porter import PorterStemmer

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, f1_score
from sklearn.feature_extraction.text import CountVectorizer


stemmer = PorterStemmer()

def clean_documents(docs):
    stemmer = PorterStemmer()
    docs_clean = docs.str.lower()
    docs_clean = docs_clean.str.replace('[^a-z\s]', '')
    docs_clean = docs_clean.apply(lambda doc: remove_stopwords(doc))
    #docs_clean = pd.Series(stemmer.stem_documents(docs_clean), index=docs.index)
    docs_clean = pd.Series(docs_clean, index=docs.index)
    return docs_clean
docs_cleaned = clean_documents(imdb['review'])
train_x, validate_x, train_y, validate_y = train_test_split(docs_cleaned,
                                                           imdb['sentiment'],
                                                           test_size=0.2,
                                                           random_state=1)
train_x.shape, validate_x.shape, train_y.shape, validate_y.shape
vectorizer = CountVectorizer(min_df=2,stop_words='english',).fit(train_x)
vocab = vectorizer.get_feature_names()
train_dtm = vectorizer.transform(train_x)
validate_dtm = vectorizer.transform(validate_x)
nb_model = MultinomialNB().fit(train_dtm, train_y)
pred_validate_y = pd.Series(nb_model.predict(validate_dtm), index=validate_y.index)
print(accuracy_score(validate_y, pred_validate_y))
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer(stop_words='english',).fit(train_x)
vocab = tfidf_vectorizer.get_feature_names()
train_dtm_tfidf = tfidf_vectorizer.transform(train_x)
validate_dtm_tfidf = tfidf_vectorizer.transform(validate_x)
nb_model_tfidf = MultinomialNB().fit(train_dtm_tfidf, train_y)
pred_validate_y = pd.Series(nb_model_tfidf.predict(validate_dtm_tfidf), index=validate_y.index)
print(accuracy_score(validate_y, pred_validate_y))
### Word Embeddings
import zipfile
import gensim
archive = zipfile.ZipFile('/kaggle/input/quora-insincere-questions-classification/embeddings.zip', 'r')
archive.namelist()

path = 'GoogleNews-vectors-negative300/GoogleNews-vectors-negative300.bin'
embeddings = gensim.models.KeyedVectors.load_word2vec_format(archive.open(path), binary=True)
#embeddings.most_similar('mercedes', topn=10)
#embeddings['india']
all_docs_vectors = pd.DataFrame()
tokens_missing = []
for doc in docs_cleaned:
    temp = pd.DataFrame()
    for token in doc.split(' '):
        try:
            word_vector = embeddings[token]
            temp = temp.append(pd.Series(word_vector), ignore_index=True)
        except:
            tokens_missing.append(token)
    
    doc_vector = temp.mean()
    all_docs_vectors = all_docs_vectors.append(doc_vector, ignore_index=True)
all_docs_vectors.head()
train_x, validate_x, train_y, validate_y = train_test_split(all_docs_vectors.fillna(0),
                                                           imdb['sentiment'],
                                                           test_size=0.2,
                                                           random_state=1)
train_x.shape, validate_x.shape, train_y.shape, validate_y.shape
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=100).fit(train_x, train_y)
validate_y_pred = rf.predict(validate_x)
print(accuracy_score(validate_y, validate_y_pred))
docs_tokens = docs_cleaned.str.split(' ').tolist()
train_x, validate_x, train_y, validate_y = train_test_split(docs_cleaned,
                                                           imdb['sentiment'],
                                                           test_size=0.2,
                                                           random_state=1)
import numpy as np
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers.embeddings import Embedding
import tensorflow as tf
vocab = set(x for l in docs_tokens for x in l)
vocab_size = len(vocab)
train_y_labels = np.array(train_y)
vocab_size = len(vocab)
max_length = max([len(x) for x in docs_tokens])
encoded_docs = [one_hot(d, vocab_size) for d in train_x]
padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2)
model = Sequential()
model.add(Embedding(vocab_size, 300, input_length=max_length))
model.add(Flatten())
model.add(Dense(8, activation='sigmoid'))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy')
model.fit(padded_docs, train_y_labels,
          epochs=3, batch_size=1000,
          validation_split=0.2,
          callbacks=[callback],
         )
encoded_docs = [one_hot(d, vocab_size) for d in validate_x]
padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
validate_y_pred = model.predict_classes(padded_docs).flatten()
accuracy_score(validate_y.values, validate_y_pred)
imdb['review'].iloc[0]
from nltk.sentiment.vader import SentimentIntensityAnalyzer

sentiment_analyzer = SentimentIntensityAnalyzer()
sentiment_analyzer.polarity_scores('they love coffee')
score = 0.5
cs = score / np.sqrt(np.square(score) + 15)
cs
## Case is important
## Punct is important
## very, not are important
## Stemming is not used
print(sentiment_analyzer.polarity_scores('they love coffee')['compound'])
print(sentiment_analyzer.polarity_scores('they LOVE coffee')['compound'])
print(sentiment_analyzer.polarity_scores('they LOVE!!! coffee')['compound'])
print(sentiment_analyzer.polarity_scores('they very LOVE!!! coffee')['compound'])
print(sentiment_analyzer.polarity_scores('they very LOVE :) coffee')['compound'])
docs = imdb.loc[validate_x.index]
docs['compound'] = docs['review'].apply(lambda v: sentiment_analyzer.polarity_scores(v)['compound'])
docs['sentiment_vader'] = docs['compound'].apply(lambda v: 1 if v >0 else 0)
accuracy_score(docs['sentiment'], docs['sentiment_vader'])
sentiment_analyzer.polarity_scores("there are better movies in youtube")