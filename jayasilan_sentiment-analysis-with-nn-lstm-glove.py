# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt
%matplotlib inline
import nltk                            # Cleaning the data
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
import os
# Load data file...

train_df = pd.read_csv('../input/word2vec-nlp-tutorial/labeledTrainData.tsv',header = 0,delimiter='\t')
test_df = pd.read_csv('../input/word2vec-nlp-tutorial/testData.tsv',header = 0,delimiter='\t')
train_df.head(10)
test_df.head(10)
print(train_df.shape)
print(test_df.shape)
train_df.isnull().sum()
test_df.isnull().sum()
test_df['Sentiment'] = test_df['id'].map(lambda x: 1 if int(x.strip('"').split('_')[1]) >=5 else 0)
y_test = test_df['Sentiment']
y_test.head(10)
test_df.drop(['Sentiment'],axis = 1,inplace = True)
test_df.head()
train_df.sentiment.value_counts()  # balanced data...
def clean_review(raw_rev):
    review_text = BeautifulSoup(raw_rev,'lxml').get_text()          # remove HTML
    review_text = re.sub('[^a-zA-Z]'," ",review_text)               # includes only words
    review_words = review_text.lower().split()              # splits words and converts it to lowercase
    
    Stop_words = set(stopwords.words("english"))                        
    
    mean_words = [w for w in review_words if not w in Stop_words]    # removes  stopwords..
    review = ' '.join(mean_words)
    
    return review
train_df['clean_review'] = train_df['review'].apply(clean_review)
test_df['clean_review'] = test_df['review'].apply(clean_review)
test_df.drop(['review'],axis = 1,inplace = True)
test_df.rename(columns = {'clean_review':'review'},inplace = True)
train_df['length_review'] = train_df['clean_review'].apply(len)
train_df.head()
test_df.head()


from keras.preprocessing import sequence,text
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Dense,Dropout,Embedding,LSTM,SpatialDropout1D,Bidirectional
from keras.utils import to_categorical
train_X = train_df.iloc[:,3].values
target = train_df.sentiment.values

from sklearn.model_selection import train_test_split
X_train, X_val, Y_train, Y_val = train_test_split( train_X, target , test_size = 0.2, random_state = 42)
print(X_train.shape,X_val.shape,Y_train.shape,Y_val.shape)
# max length of the review

r_len=[]
for text in train_df['clean_review']:
    word=word_tokenize(text)
    l=len(word)
    r_len.append(l)
    
MAX_REVIEW_LEN=np.max(r_len)
MAX_REVIEW_LEN
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

max_features = 6000
max_words = 350
batch_size = 128
epochs = 6
num_classes=1

tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(X_train))
X_train = tokenizer.texts_to_sequences(X_train)
X_val = tokenizer.texts_to_sequences(X_val)
X_train = sequence.pad_sequences(X_train, maxlen=max_words)
X_val = sequence.pad_sequences(X_val, maxlen=max_words)
X_test = tokenizer.texts_to_sequences(test_df['review'])
X_test = sequence.pad_sequences(X_test, maxlen=max_words)
print(X_train.shape,X_val.shape,X_test.shape)
test_df.head()
def get_coefs(word, *arr):
    return word, np.asarray(arr, dtype='float32')
    
def get_embed_mat(EMBEDDING_FILE, max_features,embed_dim):
    # word vectors
    embeddings_index = dict(get_coefs(*o.rstrip().rsplit(' ')) for o in open(EMBEDDING_FILE, encoding='utf8'))
    print('Found %s word vectors.' % len(embeddings_index))

    # embedding matrix
    word_index = tokenizer.word_index
    num_words = min(max_features, len(word_index) + 1)
    all_embs = np.stack(embeddings_index.values()) #for random init
    embedding_matrix = np.random.normal(all_embs.mean(), all_embs.std(), 
                                        (num_words, embed_dim))
    for word, i in word_index.items():
        if i >= max_features:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    max_features = embedding_matrix.shape[0]
    
    return embedding_matrix
EMBEDDING_FILE = '../input/glove6b/glove.6B.300d.txt'
embed_dim = 300 #word vector dim
embedding_matrix = get_embed_mat(EMBEDDING_FILE,max_features,embed_dim)
print(embedding_matrix.shape)
model = Sequential()
model.add(Embedding(max_features, embed_dim, input_length=X_train.shape[1],weights=[embedding_matrix],trainable=True))
model.add(SpatialDropout1D(0.25))
model.add(Bidirectional(LSTM(128,return_sequences=True)))
model.add(Bidirectional(LSTM(64,return_sequences=False)))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
history = model.fit(X_train, Y_train, validation_data=(X_val, Y_val),epochs=epochs, batch_size=batch_size, verbose=1)
prediction = model.predict(X_test)
y_pred = (prediction > 0.5)
from sklearn.metrics import f1_score, confusion_matrix
print('F1-score: {0}'.format(f1_score(y_pred, y_test)))
print('Confusion matrix:')
confusion_matrix(y_pred, y_test)
test_df['Sentiment'] = y_test
test_df.drop(['review'],axis = 1,inplace = True)

test_df.to_csv('Submission.csv',index = False)