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
!pip install tensorflow==2.2.0
import matplotlib.pyplot as plt

import nltk
import gensim




from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.layers import Dense, Embedding, LSTM, Conv1D, MaxPool1D
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import tensorflow as tf
tf.__version__
data = pd.read_csv('../input/fake-news-data/data.csv',index_col=0)
print(data.shape)
data.head()
data['label'] = np.where(data['label'] == 'Fake',0,1)
data['label'].value_counts().plot.bar()
data['label'].value_counts() / len(data)
X = []
stop_words = set(nltk.corpus.stopwords.words("english"))
tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
for par in data["text"].values:
    tmp = []
    sentences = nltk.sent_tokenize(par)
    for sent in sentences:
        sent = sent.lower()
        tokens = tokenizer.tokenize(sent)
        filtered_words = [w.strip() for w in tokens if w not in stop_words and len(w) > 1]
        tmp.extend(filtered_words)
    X.append(tmp)
data['text'][0]
print(X[0])
#Dimension of vectors we are generating
EMBEDDING_DIM = 100
#Creating Word Vectors by Word2Vec Method (takes time...)
w2v_model = gensim.models.Word2Vec(sentences=X, size=EMBEDDING_DIM, window=5, min_count=1)
w2v_model['covid']
tokenizer = Tokenizer()
tokenizer.fit_on_texts(X)

X = tokenizer.texts_to_sequences(X)
nos = np.array([len(x) for x in X])
len(nos[nos  < 10])
maxlen = 20
X = pad_sequences(X, maxlen = maxlen)
vocab_size = len(tokenizer.word_index) + 1


# Function to create weight matrix from word2vec gensim model
def get_weight_matrix(model, vocab):
    # total vocab size + 0 for unknown words
    vocab_size = len(vocab)+1
    #inetialize weight matrix with all zeros
    weight_matrix = np.zeros((vocab_size, EMBEDDING_DIM))
    # step vocab, store vectors using the Tokenizer's integer mapping
    for word, i in vocab.items():
        weight_matrix[i] = model[word]
    return weight_matrix


word_index = tokenizer.word_index
embedding_vectors = get_weight_matrix(w2v_model, word_index)


#create a model
dims=40
bi_model=Sequential()
bi_model.add(Embedding(vocab_size, output_dim = EMBEDDING_DIM, weights = [embedding_vectors], input_length = maxlen, trainable=False))
bi_model.add(Dropout(0.3))
bi_model.add(Bidirectional(LSTM(100))) #lstm with 100 neurons
bi_model.add(Dropout(0.3))
bi_model.add(Dense(1,activation='sigmoid'))
bi_model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])


bi_model.summary()
y = data['label']
X_train, X_test, y_train, y_test = train_test_split(X, y,random_state = 1)
history = bi_model.fit(X_train, y_train, validation_split=0.3, epochs=10)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['training', 'validation'], loc='best')
plt.show()
y_pred = (bi_model.predict(X_test) >= 0.5).astype("int")
accuracy_score(y_test, y_pred)
bi_model.save('bi_model.h5')
import pickle
with open('tokenizer.h5', 'wb') as tokenizer_file:
 
  # Step 3
  pickle.dump(tokenizer, tokenizer_file)
