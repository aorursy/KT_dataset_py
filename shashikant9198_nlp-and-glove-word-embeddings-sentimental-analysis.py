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
df = pd.read_csv("/kaggle/input/imdb-dataset-of-50k-movie-reviews/IMDB Dataset.csv")
df.head()
df.shape
# remove the stop_words from corpus to get more clean data and store in new column as review_new in same datafram
from nltk.corpus import stopwords
stop_words =stopwords.words('english')
def remove_stopwords(text):#remove_stopwords function gives stopwordfree sentences
    words=[word for word in text.split() if word not in stop_words]
    return words
df["review_new"]=df['review'].apply(lambda x: ' '.join(remove_stopwords(x)))    
df.head()   
# two classes are positive and negative, we have to classify our sentenses which give sentiment of the sentences.
lables = []
for i in df['sentiment']:
    if i == "positive":
        lables.append(1)
    else:
        lables.append(0)
Y = lables                # dependent variable Y
from keras.preprocessing.text import Tokenizer

max_words = 10000 # We will only consider the 10K most used words in this dataset
tokenizer = Tokenizer(num_words=max_words)     
tokenizer.fit_on_texts(df['review_new']) 
sequences = tokenizer.texts_to_sequences(df['review_new']) 
word_index = tokenizer.word_index
# we have to pad sentenses for same length which is 100 here
from keras.preprocessing.sequence import pad_sequences
maxlen = 100
pad = pad_sequences(sequences, maxlen=100)
pad.shape
X = pad  #independent variable X
from  keras.layers import Embedding ,Flatten, Dense
from keras.models import Sequential
# embedding_dim = 50
# model = Sequential()
# model.add(Embedding(max_words, embedding_dim, input_length=100))

# model.add(Flatten())

# model.add(Dense(32, activation='relu'))

# model.add(Dense(1, activation='sigmoid'))

# model.summary()
# model.compile(optimizer='adam',
#               loss='binary_crossentropy',metrics=['acc'])
              

# from sklearn.model_selection import train_test_split
# x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.4,random_state=50)
# print(x_test.shape)
# # print(y_test[0])
# history = model.fit(x_train, y_train,
#                     epochs=10,
#                     batch_size=32,
#                     validation_data=(x_test, y_test))
f = open('/kaggle/input/glove-global-vectors-for-word-representation/glove.6B.100d.txt')
# load pretrained GLoVe embeddings which I already have
# if your training on locl machine then you must download from GLoVe paper by stanford
# it is 100d (dimentional) embedding 
# to use embeddings we need to seperate words and their embeddings and store as key value pairs in an empty dictionary
# then after we need a mean and standard deviation of values
embeddings_index = {}
for line in f:                             
    values = line.split()
    word = values[0]
    embedding = np.asarray(values[1:], dtype='float32') # Load embedding
    embeddings_index[word] = embedding 
f.close()
# create an array of all embeddings values
#  find mean and standard deviation 


all_emb=np.stack(embeddings_index.values())
emb_mean = all_emb.mean()
emb_std = all_emb.std()
print(emb_mean)
print(emb_std)

# print(dict(list(embeddings_index.items())[0: 10]))
# embeddings_index.get('word')
#  form another matrix with same mean and standard deviation of dimentions equal to (number_of_words,embedding_dim)


embedding_dim = 100
nb_words = min(max_words, len(word_index))
print(nb_words)
emb_matrix = np.random.normal(emb_mean,emb_std,(nb_words,embedding_dim))
emb_matrix.mean(),emb_matrix.std()
for word,i in word_index.items():
    if i >= max_words:
        continue
    embedding_vector = embeddings_index.get(word)
    # If there is an embedding vector, put it in the embedding matrix
    if embedding_vector is not None: 
        emb_matrix[i] = embedding_vector
print(emb_matrix.shape)
model = Sequential()
model.add(Embedding(max_words, embedding_dim, input_length=maxlen, weights = [emb_matrix], trainable = False))
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.summary()
model.compile(optimizer='adam',
              loss='binary_crossentropy',metrics=['acc'])
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.4,random_state=50)
print(x_test.shape)
# print(y_test[0])
history = model.fit(x_train, y_train,
                    epochs=10,
                    batch_size=32,
                    validation_data=(x_test, y_test))
my_text = 'I love dogs. Dogs are the best. They are lovely, cuddly animals that only want the best for humans.'

seq = tokenizer.texts_to_sequences([my_text])
print('raw seq:',seq)
seq = pad_sequences(seq, maxlen=maxlen)
print('padded seq:',seq)
prediction = model.predict(seq)
print('positivity:',prediction)
my_text = 'The bleak economic outlook will force many small businesses into bankruptcy.'

seq = tokenizer.texts_to_sequences([my_text])
print('raw seq:',seq)
seq = pad_sequences(seq, maxlen=maxlen)
print('padded seq:',seq)
prediction = model.predict(seq)
print('positivity:',prediction)