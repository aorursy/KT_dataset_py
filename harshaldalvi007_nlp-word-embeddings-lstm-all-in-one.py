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
import pandas as pd

import numpy as np

import string

import os

import nltk

from nltk.tokenize import RegexpTokenizer

from nltk.tokenize import word_tokenize

from nltk.stem import WordNetLemmatizer

from nltk.corpus import stopwords

from sklearn.model_selection import train_test_split

import gensim

from tensorflow.python.keras.preprocessing.text import Tokenizer

from tensorflow.python.keras.preprocessing.sequence import pad_sequences



string.punctuation

stopword = nltk.corpus.stopwords.words('english')



lemmatizer = WordNetLemmatizer()

ps = nltk.PorterStemmer()

tokenizer = RegexpTokenizer(r'\w+')
maxlen = 80

batch_size = 32
data = pd.read_csv("/kaggle/input/smsspamcollectiontsv/SMSSpamCollection.tsv", sep='\t')

data.columns = ['label', 'body_text']

data.head()
data['label'] = (data['label']=='spam').astype(int)

data.head()
data.shape
def clean_text(text):

    

    ''' Text preprocessing '''



    tokens = tokenizer.tokenize(text.lower())

    

    table = str.maketrans('', '', string.punctuation)

    stripped = [w.translate(table) for w in tokens]

    tokens = [word for word in stripped if word.isalpha()]



    text = [lemmatizer.lemmatize(word) for word in tokens if word not in stopword]

    return text
data['body_text'] = data['body_text'].apply(lambda x: clean_text(x))

body_text_data = data['body_text'].values.tolist()
print(len(body_text_data))

body_text_data[0]
# word embedding using word2vec

model = gensim.models.Word2Vec(body_text_data, size=100, window=5, min_count=3)

len(model.wv.vocab)
# similarity

model.most_similar('customer')
# save model

model.wv.save_word2vec_format("spam_word2vec_model.txt", binary=False)
# Load embeddings



embeddings_index = {}

file = open(os.path.join('', 'spam_word2vec_model.txt'), encoding = "utf-8")



for record in file:

    values = record.split()

    word = values[0]

    coefficient = np.asarray(values[1:])

    embeddings_index[word] = coefficient

file.close()
len(embeddings_index)
embeddings_index['free']
tokenizer_obj = Tokenizer()

tokenizer_obj.fit_on_texts(body_text_data)

sequences = tokenizer_obj.texts_to_sequences(body_text_data)



word_index = tokenizer_obj.word_index

print("Word index", len(word_index))



X = pad_sequences(sequences, maxlen=maxlen)

print("X shape:", X.shape)



y = data['label'].values

print("y shape:", y.shape)



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=15, stratify=y)
word_index['free']
X[0], y[0]
# Create embedding matrix for words



EMBEDDING_DIM = 100



max_features = len(word_index) + 1

embedding_matrix = np.zeros((max_features, EMBEDDING_DIM))



for word, i in word_index.items():

    if i > max_features:

        continue

    embedding_vector = embeddings_index.get(word)

    if embedding_vector is not None:

        embedding_matrix[i] = embedding_vector
# embeddings for word - 'free'

embedding_matrix[9]
# Base model



from keras.models import Sequential

from keras.layers import Dense, Embedding, LSTM

from keras.layers.embeddings import Embedding

from keras.initializers import Constant



print("Build model...")



model = Sequential()



embedding_layer = Embedding(max_features, EMBEDDING_DIM, 

                            embeddings_initializer=Constant(embedding_matrix),

                            input_length=maxlen,

                            trainable=False)



model.add(embedding_layer)

model.add(LSTM(units=64, dropout=0.2, recurrent_dropout=0.2, activation='relu'))

model.add(Dense(256, activation='relu'))

model.add(Dense(1, activation='sigmoid'))



model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])



model.summary()
print("Train...")

model.fit(X_train, y_train, batch_size=batch_size, epochs=30, validation_data=(X_test, y_test), verbose=2)
score, acc = model.evaluate(X_test, y_test, batch_size=batch_size)

print('Test score:', score)

print('Test accuracy:', acc)
import matplotlib.pyplot as plt



plt.plot(model.history.history['loss'][5:])

plt.plot(model.history.history['val_loss'][5:])

plt.title('Loss over epochs')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.legend(['Train', 'Validation'], loc='best')

plt.show()