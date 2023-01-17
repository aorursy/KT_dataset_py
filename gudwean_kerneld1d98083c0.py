# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID";

 

os.environ["CUDA_VISIBLE_DEVICES"]="0";  

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt

from sklearn.metrics import classification_report, confusion_matrix,accuracy_score



from keras.preprocessing import sequence

from keras.preprocessing.text import Tokenizer

from sklearn import preprocessing

from keras.preprocessing import sequence



from keras.models import Sequential

from keras.layers import Dense, Activation, Embedding, SpatialDropout1D, MaxPooling1D, Embedding, Conv1D, Flatten, Dropout

from keras.layers import Bidirectional, GlobalMaxPool1D, LSTM

from keras.optimizers import Adam

from sklearn.utils import shuffle

from sklearn.metrics import f1_score, confusion_matrix

from sklearn.naive_bayes import MultinomialNB

from nltk.corpus import stopwords

from nltk.stem.wordnet import WordNetLemmatizer

from nltk.tokenize import word_tokenize

from nltk.stem.snowball import SnowballStemmer

from sklearn.ensemble import ExtraTreesClassifier

from sklearn.feature_extraction.text import CountVectorizer

import re



# load data

input_file = "../input/imdb-review-dataset/imdb_master.csv"



# comma delimited is the default

data = pd.read_csv(input_file, header = 0, encoding='ISO-8859-1', engine='python')
indexes_train = np.where(data['type'] == 'train')

indexes_test = np.where(data['type'] == 'test')



Xtrain = data['review'][indexes_train[0]].values

Ytrain = data['label'][indexes_train[0]].values

Xtest = data['review'][indexes_test[0]].values

Ytest = data['label'][indexes_test[0]].values



index_unsup = np.where(Ytrain == 'unsup')

Ytrain = np.delete(Ytrain, index_unsup)

Xtrain = np.delete(Xtrain, index_unsup)
from sklearn.naive_bayes import MultinomialNB

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.feature_selection import SelectKBest

from sklearn.feature_selection import f_classif

from sklearn.feature_selection import chi2

from nltk.corpus import stopwords



stopwords = set(stopwords.words("english")) 

def ngram_vectorize(train_texts, train_labels, val_texts):

    kwargs = {

        'ngram_range' : (1, 2),

        'strip_accents' : 'unicode',

        'dtype' : 'int32',

        'decode_error' : 'replace',

        'analyzer' : 'word',

        'min_df' : 1,

    }

    

    tfidf_vectorizer = TfidfVectorizer(**kwargs, stop_words = stopwords, sublinear_tf=True)

    x_train = tfidf_vectorizer.fit_transform(train_texts)

    x_val = tfidf_vectorizer.transform(val_texts)

    

    selector = SelectKBest(f_classif, k=min(6000, x_train.shape[1]))

    selector.fit(x_train, train_labels)

    x_train = selector.transform(x_train).astype('float32')

    x_val = selector.transform(x_val).astype('float32')

    return x_train, x_val
df_bag_train, df_bag_test = ngram_vectorize(Xtrain, Ytrain, Xtest)
print(df_bag_train[0])
nb = MultinomialNB()

nb.fit(df_bag_train, Ytrain)

nb_pred = nb.predict(df_bag_test)

print('Accuracy ',accuracy_score(Ytest, nb_pred))
indexes_train = np.where(data['type'] == 'train')

indexes_test = np.where(data['type'] == 'test')



X_train = data['review'][indexes_train[0]].values

Y_train = data['label'][indexes_train[0]].values



X_test = data['review'][indexes_test[0]].values

Y_test = data['label'][indexes_test[0]].values



index_unsup = np.where(Y_train == 'unsup')

Y_train = np.delete(Y_train, index_unsup)

X_train = np.delete(X_train, index_unsup)
le = preprocessing.LabelEncoder()

le.fit(Y_train)



Y_train_encod = le.transform(Y_train) 

Y_test_encod = le.transform(Y_test) 



X_train, Y_train_encod = shuffle(X_train, Y_train_encod)
max_features = 10000

maxlen = 100

embedding_dimenssion = 100



VALIDATION_SPLIT = 0.1

CLASSES = 1

NB_EPOCH = 20

BATCH_SIZE = 64

OPTIMIZER = Adam(lr=0.001)



# Tokenization and encoding text corpus

tk = Tokenizer(num_words=max_features)

tk.fit_on_texts(X_train)

X_train_en = tk.texts_to_sequences(X_train)

X_test_en = tk.texts_to_sequences(X_test)



word2index = tk.word_index

index2word = tk.index_word
X_train_new = sequence.pad_sequences(X_train_en, maxlen=maxlen)

X_test_new = sequence.pad_sequences(X_test_en, maxlen=maxlen)



glove_dir = ''.join(['../input/glove6b/glove.6B.', str(embedding_dimenssion),'d.txt'])



embeddings_index = {}



with open(glove_dir, encoding='utf-8') as f:

    for line in f:

        values = line.split()

        word = values[0]

        embedding = np.asarray(values[1:], dtype='float32')

        embeddings_index[word] = embedding 

        

print('Found {:,} word vectors in GloVe.'.format(len(embeddings_index)))
embedding_matrix = np.zeros((max_features, embedding_dimenssion))



for word, i in word2index.items():

    if i >= max_features:

        break

    embedding_vector = embeddings_index.get(word)

    if embedding_vector is not None:

        embedding_matrix[i] = embedding_vector
model = Sequential()



model.add(Embedding(max_features, embedding_dimenssion, input_length=maxlen,

                    weights=[embedding_matrix], trainable=False))

model.add(LSTM(125, dropout=0.2, recurrent_dropout=0.2))

model.add(Dense(1, activation="sigmoid"))

model.summary()





model.compile(loss='binary_crossentropy', optimizer=OPTIMIZER, metrics=['accuracy'])



model.fit(X_train_new, Y_train_encod, batch_size=BATCH_SIZE, epochs=10, validation_split=VALIDATION_SPLIT, verbose=1)



scores = model.evaluate(X_test_new, Y_test_encod)

print('losses: {}'.format(scores[0]))

print('TEST accuracy: {}'.format(scores[1]))