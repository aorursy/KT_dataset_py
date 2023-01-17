# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.corpus import stopwords                # stopwords are removed from text to keep just useful info
from nltk import word_tokenize, sent_tokenize

import keras
from keras.preprocessing.text import one_hot, Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Flatten, Embedding, LSTM, SpatialDropout1D, Input, Bidirectional,Dropout
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.regularizers import l2

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
raw_train_data_labeled = pd.read_csv("../input/word2vec-nlp-tutorial/labeledTrainData.tsv", header=0, delimiter="\t", quoting=3)
raw_train_data_unlabeled = pd.read_csv("../input/word2vec-nlp-tutorial/unlabeledTrainData.tsv", header=0, delimiter="\t", quoting=3)
raw_test_data = pd.read_csv("../input/word2vec-nlp-tutorial/testData.tsv", header=0, delimiter="\t", quoting=3)
# importing IMDB dataset from another source....this is done to increase our training dataset
# w/o this the max accuracy was around 88%, but using this validation set acc. increased to around 93%
imdb_data = pd.read_csv('../input/imdb-review-dataset/imdb_master.csv',encoding="latin-1")
imdb_data.head()

imdb_data = imdb_data.drop(["Unnamed: 0","type","file"],axis=1)
imdb_data['sentiment'] = imdb_data['label'].map({"neg":0, "pos":1})
imdb_data = imdb_data.drop(["label"],axis=1)
imdb_data = imdb_data.dropna()
imdb_data = imdb_data[['sentiment','review']]
raw_train_data_labeled.head()
raw_train_data_unlabeled.head()
X = raw_train_data_labeled['review']
y = raw_train_data_labeled['sentiment']
review_data = X.append(raw_test_data['review'])
review_data = review_data.append(imdb_data['review'])
review_data.shape
ntrain = X.shape[0]
df = raw_train_data_labeled.append(raw_test_data, sort=False)
df = df.drop(['sentiment'], axis=1)
df_review = df['review']
'''
Here we will do preprocessing
1. Removing punctuations
2. Lowering all words
3. removing non-alphabet things
4. removing stop words
5. Tokenizing the sentence
'''
import string

review_lines = list()
lines = review_data.values.tolist()

for line in lines:
    
    '''
    breaks line into it's sub parts like each word and comma etc,
    https://pythonspot.com/tokenizing-words-and-sentences-with-nltk/
    '''
    tokens = word_tokenize(line)   
    
     #convert to lower case
    tokens = [w.lower() for w in tokens]
    
    #remove punctuation from each word
    # brief detail: https://pythonadventures.wordpress.com/2017/02/05/remove-punctuations-from-a-text/
    table = str.maketrans('','', string.punctuation)
    stripped = [w.translate(table) for w in tokens]
     
    # remove remaining tokens that are not alphabetic
    words = [w for w in stripped if w.isalpha()]
    
    # filter out stop words
    stop_words = set(stopwords.words('english'))
    words = [w for w in words if w not in stop_words]
    
    review_lines.append(words)
'''
gensim is python library for training word embeddings in given data
for more information visit: 
1. https://machinelearningmastery.com/develop-word-embeddings-python-gensim/
2. http://kavita-ganesan.com/gensim-word2vec-tutorial-starter-code/#.XEoWKVwzbIV
'''
import gensim

embedding_vector_size = 150
# now training embeddings for each word 
model_1 = gensim.models.Word2Vec(sentences = review_lines, size=embedding_vector_size, min_count=1, window=5, workers=4 )

# to get total number of unique words
words = list(model_1.wv.vocab)

print("vocab size:", len(words))
#len(sequence)
leng=0
length = [(leng + len(x)) for x in review_lines]
plt.hist(length)
plt.xlabel('length of words')
plt.ylabel('frequency')
import math
avg_length = sum(length)/len(review_lines)

# if words are more than max_length then they are skipped, if less than padding with 0 is done
print(avg_length)
#max_len = math.ceil(avg_length)             # this is used to decide how man words in seq to keep
max_len = math.ceil(avg_length) 
'''
Now we have trained the embeddings, we now have embedding vector for each word. We will
convert our text training data to numeric using theseword embeddings.
First, we need to make length of each input same, therefore we'll do padding. But padding happends 
on numeric data, therefore we'll convert texts to sequences using tokenize() function. Then add padding
Then we'll replace each non-zero numeric resulted from texts to sequences to its corresponding word
embedding.
'''
max_features = 6000
tokenizer = Tokenizer(num_words=max_features)       #keeps 6000 most common words
train_test_data = review_lines                       # contains word tokens extracted from lines
tokenizer.fit_on_texts(train_test_data)
sequence = tokenizer.texts_to_sequences(train_test_data)
train_test_data = pad_sequences(sequence, maxlen = max_len)
# Preparing embedding matrix
vocab_size = len(tokenizer.word_index)+1
embedding_matrix = np.zeros((vocab_size, embedding_vector_size))
# +1 is done because i starts from 1 instead of 0, and goes till len(vocab)
for  word, i in tokenizer.word_index.items():
    embedding_vector = model_1.wv[word]
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector
X = train_test_data[:ntrain,:]
X = np.append(X,train_test_data[ntrain+25000: ,:])
X = X.reshape(-1,123)
y1 = y.append(imdb_data['sentiment'])
X_train, X_test, y_train, y_test = train_test_split(X , y1, test_size=0.2, random_state=42, shuffle=True)
model = Sequential()

model.add(Embedding(input_dim = vocab_size, output_dim = embedding_vector_size, 
                    input_length = max_len, weights = [embedding_matrix]))
model.add(Bidirectional(LSTM(64, dropout=0.25, recurrent_dropout=0.1)))
model.add(Dense(10))
model.add(Dropout(0.3))
model.add(Dense(1, activation='sigmoid'))
learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 
                                            patience=3, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.00001)
model.compile(optimizer='RMSprop', loss='binary_crossentropy', metrics=['acc'])
print(model.summary())

history = model.fit(X_train, y_train, epochs = 30, batch_size = 700, validation_data=(X_test, y_test),callbacks = [learning_rate_reduction])
import matplotlib.pyplot as plt
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
y_test_pred = model.predict(X_test)

from sklearn.metrics import roc_auc_score
roc_auc_score(y_test, y_test_pred, average = 'weighted')
#predicting test_data
y_pred = model.predict(train_test_data[ntrain:ntrain+25000 , :])
predictions = [1 if (x>0.5) else 0 for x in y_pred ]
predictions = pd.Series(predictions)
ids = raw_test_data['id'].str.replace('"', '')
submission = pd.DataFrame({'id': ids, 'sentiment':predictions})
submission.to_csv('word2vec.csv',index=False)
