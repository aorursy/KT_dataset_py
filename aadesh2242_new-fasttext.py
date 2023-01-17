#importing standard libraries

import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt



from sklearn.model_selection import train_test_split

#Importing Keras so that I can apply CNN

import keras

from keras import optimizers

from keras import backend as K

from keras import regularizers

from keras.models import Sequential

from keras.layers import Dense, Activation, Dropout, Flatten

from keras.layers import Embedding, Conv1D, MaxPooling1D, GlobalMaxPooling1D 

from keras.utils import plot_model

from keras.preprocessing import sequence

from keras.preprocessing.text import Tokenizer

from keras.callbacks import EarlyStopping



#importing libraries for data processing

from tqdm import tqdm

from nltk.corpus import stopwords

from nltk.tokenize import RegexpTokenizer 

import os, re, csv, math, codecs



sns.set_style("whitegrid")

np.random.seed(0)

#Setting path for kaggle data.

DATA_PATH = '../input/'

EMBEDDING_DIR = '../input/'



MAX_NB_WORDS = 100000

tokenizer = RegexpTokenizer(r'\w+')

stop_words = set(stopwords.words('english'))

stop_words.update(['.', ',', '"', "'", ':', ';', '(', ')', '[', ']', '{', '}'])
len(stop_words)
stop_words_2 = set(stopwords.words('spanish'))
stop_words_2
#load embeddings

embeddings_index = {}

#Have downloaded fastext pretrained embeddings. So loading it here

f = codecs.open('../input/fasttext2/wiki.simple.vec', encoding='utf-8')

print(f)

for line in tqdm(f):

    values = line.rstrip().rsplit(' ')

    word = values[0]

    #print(word)

    coefs = np.asarray(values[1:], dtype='float32')

    embeddings_index[word] = coefs

f.close()
train_df = pd.read_csv('../input/jigsaw-toxic-comment-classification-challenge' + '/train.csv', sep=',', header=0)
train_df.head()
label_names = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
# separate explanatory and dependent variables

y = train_df[label_names].values
len(y)
raw_docs_train = train_df['comment_text'].tolist()

# split for cross-validation (train-60%, validation 20% and test 20%)
len(raw_docs_train)
num_classes = len(label_names)



print(num_classes)
processed_docs_train = []

for doc in tqdm(raw_docs_train):

    tokens = tokenizer.tokenize(doc)

    filtered = [word for word in tokens if word not in stop_words]

    processed_docs_train.append(" ".join(filtered))
raw_docs_train[0]
processed_docs_train[0]
#Tokenizing data.



tokenizer_train = Tokenizer(num_words=MAX_NB_WORDS, lower=True, char_level=False)

tokenizer_train.fit_on_texts(processed_docs_train)  #leaky
type(tokenizer_train)
word_seq_train = tokenizer_train.texts_to_sequences(processed_docs_train)

word_index = tokenizer_train.word_index

print("dictionary size: ", len(word_index))
len(word_seq_train)
train_df['doc_len'] = train_df['comment_text'].apply(lambda words: len(words.split(" ")))

max_seq_len = np.round(train_df['doc_len'].mean() + train_df['doc_len'].std()).astype(int)
train_df.head()
max_seq_len
train_df['doc_len'].mean()
   #So here we are padding the sequnce so that I will work ML algorithms because

    #it should have same length

word_seq_train = sequence.pad_sequences(word_seq_train, maxlen=max_seq_len)



X_train, X_test, y_train, y_test = train_test_split(word_seq_train, y, test_size=0.4, random_state=123)



X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5, random_state=123)

word_seq_train[0]
processed_docs_train[0]
word_index
word_seq_train.shape
#training params

batch_size = 256 

num_epochs = 20 



#model parameters

num_filters = 64 

embed_dim = 300 

weight_decay = 1e-4
#embedding matrix

words_not_found = []

nb_words = min(MAX_NB_WORDS, len(word_index))

embedding_matrix = np.zeros((nb_words, embed_dim))
embedding_matrix.shape
for word, i in word_index.items():

    if i >= nb_words:

        continue

    embedding_vector = embeddings_index.get(word)

    if (embedding_vector is not None) and len(embedding_vector) > 0:

        # words not found in embedding index will be all-zeros.

        embedding_matrix[i] = embedding_vector

    else:

        words_not_found.append(word)

print('number of null word embeddings: %d' % np.sum(np.sum(embedding_matrix, axis=1) == 0))
print("Words not found in the embedding: ", np.random.choice(words_not_found, 50))
len(word_seq_train)
len(words_not_found)
embedding_vector
#CNN model training

model = Sequential()

model.add(Embedding(nb_words, embed_dim,

          weights=[embedding_matrix], input_length=max_seq_len, trainable=False))

model.add(Conv1D(num_filters, 7, activation='relu', padding='same'))

model.add(MaxPooling1D(2))

model.add(Conv1D(num_filters, 7, activation='relu', padding='same'))

model.add(GlobalMaxPooling1D())

model.add(Dropout(0.5))

model.add(Dense(32, activation='relu', kernel_regularizer=regularizers.l2(weight_decay)))

model.add(Dense(num_classes, activation='sigmoid'))  #multi-label (k-hot encoding)



adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])

model.summary()
early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.01, patience=4, verbose=1)

callbacks_list = [early_stopping]
X_train[0]
#sending the data to CNN to train the model

#hist = model.fit(word_seq_train, y_train, batch_size=batch_size, epochs=num_epochs, callbacks=callbacks_list, validation_split=0.1, shuffle=True, verbose=2)

hist = model.fit(X_train, y_train, batch_size=batch_size, epochs=num_epochs, callbacks=callbacks_list, validation_data = (X_val,y_val), shuffle=True, verbose=2)


# Predict on train, val and test datasets

pred_train = model.predict(X_train)
pred_test = model.predict(X_test)

pred_val = model.predict(X_val)



AUC = np.zeros((3,6))

AUC
from sklearn import metrics

for i,x in enumerate(label_names):

    auc = np.array([metrics.roc_auc_score(y_train[:,i], pred_train[:,i]),

                    metrics.roc_auc_score(y_val[:,i], pred_val[:,i]),

                    metrics.roc_auc_score(y_test[:,i], pred_test[:,i])])

    print(x,"Train AUC:",auc[0],", Val AUC:",auc[1],", Test AUC:",auc[2])

    AUC[:,i] = auc

    

avg_auc = AUC.mean(axis=1)

print("Average Train AUC:",avg_auc[0],", Average Val AUC:",avg_auc[1],", Average Test AUC:",avg_auc[2])
plt.figure()

plt.plot(hist.history['acc'], lw=2.0, color='b', label='train')

plt.plot(hist.history['val_acc'], lw=2.0, color='r', label='val')

plt.title('CNN sentiment')

plt.xlabel('Epochs')

plt.ylabel('Accuracy')

plt.legend(loc='upper left')

plt.show()