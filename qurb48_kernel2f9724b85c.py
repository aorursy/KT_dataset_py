import sys, os, re, csv, codecs, numpy as np, pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, SpatialDropout1D, Conv1D

from tensorflow.keras.layers import Bidirectional, GlobalMaxPool1D,Bidirectional, concatenate,GlobalAvgPool1D

from tensorflow.keras.models import Model

from tensorflow.keras import initializers, regularizers, constraints, optimizers, layers

import matplotlib.pyplot as plt

%matplotlib inline

import gensim.models.keyedvectors as word2vec

import gc
path = '../input/'

comp = 'jigsaw-toxic-comment-classification-challenge/'

# EMBEDDING_FILE=f'{path}glove6b50d/glove.6B.50d.txt'

# EMBEDDING_FILE = f'{path}fasttext-crawl-300d-2m/crawl-300d-2M.vec'

TRAIN_DATA_FILE=f'{path}{comp}train.csv.zip'

TEST_DATA_FILE=f'{path}{comp}test.csv.zip'

TEST_LABELS_FILE = f'{path}{comp}test_labels.csv.zip'
train = pd.read_csv(TRAIN_DATA_FILE)

test = pd.read_csv(TEST_DATA_FILE)

test_l = pd.read_csv(TEST_LABELS_FILE)

sample_submission = pd.read_csv(f'{path}{comp}sample_submission.csv.zip')
embed_size = 300 # how big is each word vector

max_features = 20000 # how many unique words to use (i.e num rows in embedding vector)

maxlen = 200 # max number of words in a comment to use
train.isnull().any(),test.isnull().any()
def clean_text(text):

    text = text.lower()

    text = re.sub(r"what's", "what is ", text)

    text = re.sub(r"\'s", " ", text)

    text = re.sub(r"\'ve", " have ", text)

    text = re.sub(r"can't", "cannot ", text)

    text = re.sub(r"n't", " not ", text)

    text = re.sub(r"i'm", "i am ", text)

    text = re.sub(r"\'re", " are ", text)

    text = re.sub(r"\'d", " would ", text)

    text = re.sub(r"\'ll", " will ", text)

    text = re.sub(r"\'scuse", " excuse ", text)

    text = re.sub('\W', ' ', text)

    text = re.sub('\s+', ' ', text)

    text = re.sub(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', ' _ip_ ', text) # Replace ips

    text = re.sub(r'([\'\"\.\(\)\!\?\-\\\/\,])', r' \1 ',text) # Isolate punctuation

    text = re.sub(r'([\;\:\|•«\n])', ' ', text) # Remove some special characters

    text = text.replace('&', ' and ') # Replace numbers and symbols with language

    text = text.replace('@', ' at ')

    text = text.replace('0', ' zero ')

    text = text.replace('1', ' one ')

    text = text.replace('2', ' two ')

    text = text.replace('3', ' three ')

    text = text.replace('4', ' four ')

    text = text.replace('5', ' five ')

    text = text.replace('6', ' six ')

    text = text.replace('7', ' seven ')

    text = text.replace('8', ' eight ')

    text = text.replace('9', ' nine ')

    text = text.strip(' ')

    return text
train['comment_text'] = train['comment_text'].map(lambda com : clean_text(com))
print(train.comment_text.shape, test.comment_text.shape)
# to do

list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

y = train[list_classes].values

list_sentences_train = train["comment_text"]

list_sentences_test = test["comment_text"]
tokenizer = Tokenizer(num_words=max_features)

tokenizer.fit_on_texts(list(list_sentences_train))

list_tokenized_train = tokenizer.texts_to_sequences(list_sentences_train)

list_tokenized_test = tokenizer.texts_to_sequences(list_sentences_test)



X_t = pad_sequences(list_tokenized_train, maxlen=maxlen)

X_te = pad_sequences(list_tokenized_test, maxlen=maxlen)
comp = 'crawl300d2m/'

embedding_path = f'{path}{comp}crawl-300d-2M.vec'
def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')

embedding_index = dict(get_coefs(*o.strip().split(" ")) for o in open(embedding_path))
word_index = tokenizer.word_index

nb_words = min(max_features, len(word_index))

embedding_matrix = np.zeros((nb_words, embed_size))

for word, i in word_index.items():

    if i >= max_features: continue

    embedding_vector = embedding_index.get(word)

    if embedding_vector is not None: embedding_matrix[i] = embedding_vector
embedding_matrix
import tensorflow as tf

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Flatten, Dense, Dropout, Activation, LSTM, Bidirectional, GlobalMaxPool1D, Embedding,AveragePooling1D

from tensorflow.keras import Input, Model

from tensorflow.keras import layers, initializers, regularizers, constraints, optimizers

from tensorflow.keras.callbacks import EarlyStopping

from tensorflow.keras.callbacks import ModelCheckpoint



from tensorflow.compat.v1.keras.layers import CuDNNGRU, CuDNNLSTM
from tensorflow.keras.layers import LeakyReLU

from keras.layers import GRU
x=0

model = 0
maxlen=200



inp = Input(shape=(maxlen,)) # max_len = 100



x = Embedding(max_features, embed_size, weights=[embedding_matrix])(inp) # max_features = 20000, embed_size = 50 / embedding_matrix ?



x= SpatialDropout1D(0.2)(x)



x = Bidirectional(CuDNNLSTM(128, return_sequences=True))(x)



x = Conv1D(64, kernel_size = 3, padding = "valid", kernel_initializer = "glorot_uniform")(x)



avg_pool = GlobalAvgPool1D()(x)

max_pool = GlobalMaxPool1D()(x)



conc = concatenate([avg_pool,max_pool])



x = Dense(50)(conc)

x = LeakyReLU(alpha = 0.01)(x)



x = Dropout(0.25)(x)



x = Dense(6, activation = "sigmoid")(x)

model = Model(inputs=inp, outputs=x)



model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])



model.summary()
model = Model(inputs=inp, outputs=x)

model.compile(loss='binary_crossentropy',

                  optimizer='adam',

                  metrics=['accuracy'])
#모델 저장



file_path = "bi_gru.best.hdf5"

checkpoint = ModelCheckpoint(file_path, monitor='val_loss',verbose=1, save_best_only=True, mode = 'min')



early = EarlyStopping(monitor="val_loss", mode="min",patience=1)
X_t.shape
#NLP에서 epoch 수 많이 하면 overfitting의 문제 생길 가능성 높음

batch_size = 32

epochs = 10 

callbacks_list = [checkpoint,early]

history = model.fit(X_t,y, batch_size=batch_size, epochs=epochs, validation_split=0.1, callbacks=callbacks_list)
acc = history.history['accuracy']

val_acc = history.history['val_accuracy']



loss = history.history['loss']

val_loss = history.history['val_loss']



plt.figure(figsize=(8, 8))

plt.subplot(2, 1, 1)

plt.plot(acc, label='Training Accuracy')

plt.plot(val_acc, label='Validation Accuracy')

plt.legend(loc='lower right')

plt.ylabel('Accuracy')

plt.ylim([min(plt.ylim()),1])

plt.title('Training and Validation Accuracy')



plt.subplot(2, 1, 2)

plt.plot(loss, label='Training Loss')

plt.plot(val_loss, label='Validation Loss')

plt.legend(loc='upper right')

plt.ylabel('Cross Entropy')

plt.ylim([0,max(plt.ylim())])

plt.title('Training and Validation Loss')

plt.show()
from sklearn.metrics import accuracy_score



batch_size = 1024

results = model.predict(X_te, batch_size=batch_size,verbose = 1)
sample_submission[list_classes] = results

sample_submission.to_csv('fin.csv', index=False)



files.download('fin.csv')