#Loading Required Packages

from keras.utils import np_utils

import os 
import pandas as pd
import numpy as np
import re

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize,sent_tokenize
from nltk.stem import WordNetLemmatizer

from keras.models import Sequential
from keras.preprocessing.text import Tokenizer
from keras.layers import Dense, LSTM, Activation, BatchNormalization, Bidirectional 

from keras.utils import to_categorical
from keras import optimizers
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.preprocessing.sequence import pad_sequences



from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report


restaurant_review = pd.read_csv("Restaurant_Reviews.tsv",delimiter = '\t')


restaurant_review
def review_to_words( raw_review ):
        
    letters_only = re.sub("[^a-zA-Z]", " ", raw_review) 
    words = letters_only.lower().split()                             
    stops = set(stopwords.words("english"))                   
    meaningful_words = [w for w in words if not w in stops]   
    return( " ".join( meaningful_words )) 
#import spacy

num_reviews = restaurant_review['Review'].size
review_to_words(restaurant_review['Review'][0])
# Initialize an empty list to hold the clean reviews
clean_reviews = []

# Loop over each review; create an index i that goes from 0 to the length
# of the movie review list 
for i in range( 0, num_reviews ):
    # Call our function for each one, and add the result to the list of
    # clean reviews
    clean_reviews.append(review_to_words( restaurant_review['Review'][i] ) )

clean_reviews
X = clean_reviews
y = restaurant_review.iloc[:, 1].values
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)
clean_reviews
#Binarize labels for neural networks
#y_train_en = np_utils.to_categorical(y_train)
#y_test_en = np_utils.to_categorical(y_test)
np.unique(y_train)
t = Tokenizer()
t.fit_on_texts(X_train)



vocab_size = len(t.word_index) + 1
print (vocab_size)

#Sequence
encoded_docs_train = t.texts_to_sequences(X_train)




#Padding
max_length = 70
X_train = pad_sequences(encoded_docs_train, maxlen=max_length)
print(X_train)

word_index = t.word_index

t1 = Tokenizer()
t1.fit_on_texts(X_test)



vocab_size1 = len(t.word_index) + 1
print (vocab_size1)

#Sequence
encoded_docs_test = t.texts_to_sequences(X_test)


#Padding
max_length = 70
X_test = pad_sequences(encoded_docs_test, maxlen=max_length)
print(X_test)




# Initializing Count Vectorizer - #tekenizxes and also count the number of occurences
#vectorizer = CountVectorizer(analyzer = "word",min_df=2) 
# fit_transform() does two functions: First, it fits the model and learns the vocabulary; 
# second, it transforms our training data into feature vectors. The input to fit_transform should be a list of strings.
#data_features = vectorizer.fit_transform(clean_reviews)
#print(data_features)
#data_features.shape
#model = Word2Vec(clean_reviews, size=100, window=5, min_count=5, workers=4)
len(X_train)
y_train
# load the GloVe vectors in a dictionary:
from tqdm import tqdm

embeddings_index = {}
f = open('glove.6B.100d.txt',encoding= "latin1")
for line in tqdm(f):
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' % len(embeddings_index))



# create an embedding matrix for the words we have in the dataset
embedding_matrix = np.zeros((len(word_index) + 1, 100))
for word, i in tqdm(word_index.items()):
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector
embedding_matrix
model = Sequential()
model.add(Embedding(len(word_index) + 1,
                     100,
                     weights=[embedding_matrix],
                     input_length=max_length,
                     trainable=False))

model.add(LSTM(100, dropout=0.3, recurrent_dropout=0.3))
model.add(Dense(1024, activation='relu'))
model.add(Dense(1024, activation='relu'))
model.add(Dense(3))
model.add(Activation('sigmoid'))
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(X_train, y=y_train, batch_size=50, epochs=20, verbose=1)
train_pred = model.predict_classes(X_train)
test_pred = model.predict_classes(X_test)
test_pred
from sklearn.metrics import confusion_matrix, roc_curve, auc

confusion_matrix_test = confusion_matrix(y_test, test_pred)
confusion_matrix_train = confusion_matrix(y_train, train_pred)

valid_acc = accuracy_score(y_test, test_pred)
train_acc = accuracy_score(y_train, train_pred)


print(valid_acc)
print(train_acc)

#with Modified Xavier initializer
model1 = Sequential()
model1.add(Embedding(len(word_index) + 1,
                     100,
                     weights=[embedding_matrix],
                     input_length=max_length,
                     trainable=False))

model1.add(Bidirectional(LSTM(300, dropout=0.3, recurrent_dropout=0.3)))

model1.add(BatchNormalization())
model1.add(Dense(1024, activation='relu',kernel_initializer= "glorot_normal"))

#model.add(Dropout(0.8))

model1.add(Dense(1024, activation='relu'))
#model.add(Dropout(0.8))

model1.add(Dense(3))
model1.add(Activation('softmax'))
model1.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['acc'])
model1.fit(X_train, y=y_train, batch_size=50, epochs=20, verbose=1)
train_pred1 = model1.predict_classes(X_train)
test_pred1 = model1.predict_classes(X_test)
confusion_matrix_test1 = confusion_matrix(y_test, test_pred1)
confusion_matrix_train1 = confusion_matrix(y_train, train_pred1)

valid_acc1 = accuracy_score(y_test, test_pred1)
train_acc1 = accuracy_score(y_train, train_pred1)


print(valid_acc1)
print(train_acc1)