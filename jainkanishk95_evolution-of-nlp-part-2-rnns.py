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
#!pip install tensorflow-gpu
import random

import os

import numpy as np

def seed_everything(seed=1234):

    random.seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)

    np.random.seed(seed)

    #torch.manual_seed(seed)

    #torch.cuda.manual_seed(seed)

    #torch.backends.cudnn.deterministic = True

    #from tensorflow import set_random_seed

    #set_random_seed(2)



seed_everything()
#!pip install tensorflow-gpu
import tensorflow as tf

print(tf.test.gpu_device_name())

# See https://www.tensorflow.org/tutorials/using_gpu#allowing_gpu_memory_growth

#config = tf.ConfigProto()

#config.gpu_options.allow_growth = True
x = tf.random.uniform([3, 3])



print("Is there a GPU available: "),

print(tf.test.is_gpu_available())



print("Is the Tensor on GPU #0:  "),

print(x.device.endswith('GPU:0'))



print("Device name: {}".format((x.device)))
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from sklearn.feature_extraction.text import CountVectorizer

from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from keras.models import Sequential

from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D, Dropout, Bidirectional, BatchNormalization, Conv1D, GlobalAveragePooling1D 

from sklearn.model_selection import train_test_split

from keras.utils.np_utils import to_categorical

import re
train_dataset = pd.read_csv("/kaggle/input/steam-reviews/train.csv", delimiter=",")

train_dataset
test_dataset = pd.read_csv("/kaggle/input/steam-reviews-test-dataset/test.csv", delimiter=",")

test_dataset['user_suggestion'] = None
dataset = pd.concat([train_dataset, test_dataset], axis = 0)

dataset.reset_index(drop = True, inplace = True)

dataset
import re

def decontracted(phrase):

    # specific

    phrase = re.sub(r"won\'t", "will not", phrase)

    phrase = re.sub(r"can\'t", "can not", phrase)



    # general

    phrase = re.sub(r"n\'t", " not", phrase)

    phrase = re.sub(r"\'re", " are", phrase)

    phrase = re.sub(r"\'s", " is", phrase)

    phrase = re.sub(r"\'d", " would", phrase)

    phrase = re.sub(r"\'ll", " will", phrase)

    phrase = re.sub(r"\'t", " not", phrase)

    phrase = re.sub(r"\'ve", " have", phrase)

    phrase = re.sub(r"\'m", " am", phrase)

    phrase = re.sub(r"early access review", "early access review ", phrase)

    phrase = re.sub(r"\+", " + ", phrase) 

    phrase = re.sub(r"\-", " - ", phrase)     

    phrase = re.sub(r"/10", "/10 ", phrase)     

    phrase = re.sub(r"10/", " 10/", phrase)         

    return phrase
import re

def clean_reviews(lst):

    # remove URL links (httpxxx)

    lst = np.vectorize(remove_pattern)(lst, "https?://[A-Za-z0-9./]*")

    # remove special characters, numbers, punctuations (except for #)

    lst = np.core.defchararray.replace(lst, "[^a-zA-Z]", " ")

    # remove amp with and

    lst = np.vectorize(replace_pattern)(lst, "amp", "and")  

    # remove hashtags

    lst = np.vectorize(remove_pattern)(lst, "#[A-Za-z0-9]+")

    lst = np.vectorize(remove_pattern)(lst, "#[\w]*")    

    return lst

def remove_pattern(input_txt, pattern):

    r = re.findall(pattern, input_txt)

    for i in r:

        input_txt = re.sub(i, '', input_txt)        

    return input_txt

def replace_pattern(input_txt, pattern, replace_text):

    r = re.findall(pattern, input_txt)

    for i in r:

        input_txt = re.sub(i, replace_text, input_txt)        

    return input_txt
# Applying pre-processing to user reviews

text2 = clean_reviews(list(dataset['user_review'].astype('str')))

text3 = [ta.lower() for ta in text2]

text4 = [''.join([i if ord(i) < 128 else ' ' for i in t]) for t in text3]

text5 = [decontracted(u) for u in text4]

text5[1]
dataset2 = dataset[['user_review', 'user_suggestion']]

dataset2['user_review'] = text5

dataset2
dataset3 = dataset2.iloc[:17494,]

dataset3
max_words = 15000

max_len = 400

tokenizer = Tokenizer(num_words=max_words, split=' ')

tokenizer.fit_on_texts(dataset3['user_review'].values)

word_index = tokenizer.word_index

print('Found %s unique tokens.' % len(word_index))
X = tokenizer.texts_to_sequences(dataset3['user_review'].values)

X = pad_sequences(X, max_len)

X[1,], dataset3.loc[1,'user_review']
Y = pd.get_dummies(dataset3['user_suggestion']).values

#Y = dataset3['user_suggestion'].values

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2, random_state = 42)

print(X_train.shape,Y_train.shape)

print(X_test.shape,Y_test.shape)
embed_dim = 100

lstm_out = 128



model = Sequential()
model.add(Embedding(max_words, embed_dim, input_length = max_len))
model.add(LSTM(lstm_out))

model.add(Dropout(0.5))

model.add(Dense(64,activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(2,activation='softmax'))
import numpy as np

from keras.callbacks import Callback

from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score



class Metrics(Callback):

    def on_train_begin(self, logs={}):

        self.val_f1s = []

        self.val_recalls = []

        self.val_precisions = []



    def on_epoch_end(self, epoch, logs={}):

        val_predict = (np.asarray(self.model.predict(self.validation_data[0]))).round()

        val_targ = self.validation_data[1]

        _val_f1 = f1_score(val_targ, val_predict, average = "weighted")

        _val_recall = recall_score(val_targ, val_predict)

        _val_precision = precision_score(val_targ, val_predict)

        self.val_f1s.append(_val_f1)

        self.val_recalls.append(_val_recall)

        self.val_precisions.append(_val_precision)

        #print “ — val_f1: %f — val_precision: %f — val_recall %f” %(_val_f1, _val_precision, _val_recall)

        print(' — val_f1: %f — val_precision: %f — val_recall %f' % (_val_f1, _val_precision, _val_recall))

        return



metrics = Metrics()
#from sklearn.metrics import f1_score
model.compile(loss = 'binary_crossentropy', optimizer='adam', metrics = ['accuracy'])

print(model.summary())
batch_size = 256



model.fit(X_train, Y_train, epochs = 10, batch_size=batch_size, verbose = 1, validation_split = 0.2)
pred = np.argmax(model.predict(X_test), axis = 1)

actual = np.argmax(Y_test, axis = 1)



pred, actual
from sklearn.metrics import accuracy_score

acc = accuracy_score(actual, pred)

print("Accuracy of LSTM  is {}".format(acc))
glove_dir = '../input/glove-global-vectors-for-word-representation/'

embeddings_index = {}

f = open(os.path.join(glove_dir, 'glove.6B.200d.txt'))



for line in f:

    values = line.split()

    word = values[0]

    coefs = np.asarray(values[1:], dtype='float32')

    embeddings_index[word] = coefs

f.close()

print('Found %s word vectors.' % len(embeddings_index))
embedding_dim = 200

embedding_matrix = np.zeros((max_words, embedding_dim))

for word, i in word_index.items():

    if i < max_words:

        embedding_vector = embeddings_index.get(word)

        if embedding_vector is not None:

            embedding_matrix[i] = embedding_vector
tf.keras.backend.clear_session()
model = Sequential()

model.add(Embedding(max_words, embedding_dim, input_length = max_len, weights = [embedding_matrix]))

#model.add(Conv1D(128, 5, activation = 'relu'))

#model.add(GlobalAveragePooling1D())

model.add(SpatialDropout1D(0.4))

model.add(Bidirectional(LSTM(64, return_sequences=True)))

model.add(BatchNormalization())

model.add(Bidirectional(LSTM(64, return_sequences=True)))

model.add(BatchNormalization())

model.add(Bidirectional(LSTM(64)))

model.add(BatchNormalization())

#model.add(GlobalAveragePooling1D())

model.add(Dense(64,activation='relu'))

model.add(BatchNormalization())

model.add(Dense(32,activation='relu'))

model.add(BatchNormalization())

model.add(Dense(2,activation='softmax'))

model.compile(loss = 'categorical_crossentropy', optimizer='rmsprop',metrics = ['accuracy'])

print(model.summary())
tf.random.set_seed(123)

np.random.seed(123)
batch_size = 256



model.fit(X_train, Y_train, epochs = 20, batch_size=batch_size, verbose = 1, validation_split = 0.2)
pred = np.argmax(model.predict(X_test), axis = 1)

actual = np.argmax(Y_test, axis = 1)



pred, actual
from sklearn.metrics import accuracy_score

acc = accuracy_score(actual, pred)

print("Accuracy of LSTM  is {}".format(acc))