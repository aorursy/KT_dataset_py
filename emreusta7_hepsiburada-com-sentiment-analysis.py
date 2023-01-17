import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfTransformer #use TFIDF transformer to change text vector created by count vectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm,naive_bayes
from sklearn.svm import SVC #Support Vector Machine
from sklearn.metrics import accuracy_score

import tensorflow as tf
# import tensorflow_datasets as tfds
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense,Embedding,Flatten

#!pip install install ITU-Turkish-NLP-Pipeline-Caller
#import pipeline_caller
#caller = pipeline_caller.PipelineCaller()

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
hb = pd.read_csv("../input/hepsiburada/hepsiburada.csv")
hb.shape
hb.head(5)
hb['Rating'].value_counts()
hb['Review'].dropna(inplace=True)
hb['Review'] = [entry.lower() for entry in hb['Review']]
rating = hb["Rating"].values.tolist()
review = hb["Review"].values.tolist()

review_train, review_test, rating_train, rating_test = train_test_split(review,rating,test_size=0.3)

print(len(review_train))
print(len(review_test))
tokenizer = Tokenizer(num_words = 10000)
review_tokens = tokenizer.fit_on_texts(review)
review_train_tokens = tokenizer.fit_on_texts(review_train)
review_test_tokens = tokenizer.fit_on_texts(review_test)
word_index = tokenizer.word_index
word_index
review_seq = tokenizer.texts_to_sequences(review)
review_train_seq = tokenizer.texts_to_sequences(review_train)
review_test_seq = tokenizer.texts_to_sequences(review_test)
print(review[18])
print(review_seq[18])
word_count = tokenizer.word_counts
word_count = pd.DataFrame.from_dict(word_count,orient='index') # Turning the list into a dataframe
word_count.columns = ['freq']
word_count.sort_values(by=['freq'],ascending = False)
tokens_count = [len(tokens) for tokens in review_seq]
tokens_count = np.array(tokens_count)
max_token = np.max(tokens_count)
max_index = np.argmax(tokens_count)
print("token count of review that has maxium number of token: ",max_token)
print("\nindex of review that has maximum number of token: ",max_index)
print("\n",review[max_index])
review_train_pad = pad_sequences(review_train_seq, maxlen=max_token)
review_test_pad = pad_sequences(review_test_seq, maxlen=max_token)
review_pad = pad_sequences(review_seq, maxlen = max_token)

print(review_train_pad.shape)
print(np.array(review_train_seq[18]))
print(review_train_pad[18])
count_vect = CountVectorizer(max_features = 1000)
review_train_vect = count_vect.fit_transform(review_train)
review_test_vect = count_vect.fit_transform(review_test)
tfidf_vect = TfidfVectorizer(max_features = 1000)
tfidf_vect.fit(review_train)
tfidf_review_train = tfidf_vect.transform(review_train)
tfidf_review_test = tfidf_vect.transform(review_test)
print(tfidf_review_train)
SVM = svm.SVC(C =1.0, kernel ='linear', degree =3, gamma ='auto')
SVM.fit(tfidf_review_train,review_train)
#predictions_SVM = SVM.predict(tfidf_review_test)
#print("SVM Accuracy Score -> ",accuracy_score(predictions_SVM, Test_Y)*100)
nb = naive_bayes.MultinomialNB()
nb.fit(tfidf_review_train,review_train)
predictions_NB = nb.predict(tfidf_review_test)
print("Naive Bayes Accuracy Score -> ",accuracy_score(predictions_NB, Test_Y)*100)
model = Sequential([
    Embedding(
      input_dim = num_words,
      output_dim = 50, 
      input_length = max_token),
    Dense(1, activation="sigmoid")])
model.compile(
    loss="binary_crossentropy",
    optimizer='adam',
    metrics=["accuracy"])
model.summary()
model.fit(review_train_pad, np.array(rating_train), epochs=10, batch_size=256)