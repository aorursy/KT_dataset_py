# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import tensorflow as tf

import os



# Any results you write to the current directory are saved as output.
train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")

train.head(10)
train.label.value_counts()

train.label.value_counts().plot(kind='bar')
train.text.str.split().apply(len).quantile(0.99)
# Max number of words to be used

MAX_NB_WORDS = 10*1024

# Max number of words in each complaint (you can also change this)

MAX_SEQUENCE_LENGTH = 30 



# One hot encode the labels too

labels = ['normal', 'sarcastic']



def process_text(train, test):

    # you might want to do some text cleaning

    

    # you might want to stem words

    

    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=MAX_NB_WORDS, filters='', lower=True)

    tokenizer.fit_on_texts(train.text) # only fit on train data

    

    # print number of words  found/used

    word_index = tokenizer.word_index

    print('Found %s unique tokens.' % len(word_index))

    

    # tokenizer the train text into words and create the enumeration

    X_train = tokenizer.texts_to_sequences(train.text)

    # pad tweets that are smaller with zero

    X_train = tf.keras.preprocessing.sequence.pad_sequences(X_train, maxlen=MAX_SEQUENCE_LENGTH)

    

    x_train_other = train.loc[:,"accountAge":"#uppercases"].astype(float)

    

    # tokenizer the test text into words and create the enumeration

    X_test = tokenizer.texts_to_sequences(test.text)

    # pad tweets that are smaller with zero

    X_test = tf.keras.preprocessing.sequence.pad_sequences(X_test, maxlen=MAX_SEQUENCE_LENGTH)

    X_test_other = test.loc[:,"accountAge":"#uppercases"].astype(float)

    

    # One hot encode the labels too

    y_train = pd.get_dummies(train.label)[labels]

    

    print('Shape of train data tensor:', X_train.shape)

    print('Shape of train label tensor:', y_train.shape)

    print('Shape of test data tensor:', X_test.shape)

    

    return X_train, x_train_other, y_train, X_test, X_test_other, word_index

    



X_train, X_train_other,  y_train, X_test, X_test_other, word_index = process_text(train, test)









    
X_train_other.shape
def pretrained_embeddings(file_path, EMBEDDING_DIM, MAX_NB_WORDS, word2idx):

    # 1.load in pre-trained word vectors     #feature vector for each word  

    word2vec = {}

    with open(os.path.join(file_path),  errors='ignore', encoding='utf8') as f:

        # is just a space-separated text file in the format:

        # word vec[0] vec[1] vec[2] ...

        for line in f:

            values = line.split()

            word = values[0]

            vec = np.asarray(values[1:], dtype='float32')

            word2vec[word] = vec



        print('Found %s word vectors.' % len(word2vec))



    # 2.prepare embedding matrix

    print('Filling pre-trained embeddings...')

    num_words = MAX_NB_WORDS

    # initialization by zeros

    embedding_matrix = np.random.random((num_words, EMBEDDING_DIM))

    for word, i in word2idx.items():

        if i < num_words:

            embedding_vector = word2vec.get(word)

            if embedding_vector is not None:

              # words not found in embedding index will be all zeros.

              embedding_matrix[i] = embedding_vector



    embedding_matrix[0]  = 0.0

    return embedding_matrix





EMBEDDING_DIM = 50

file = "../input/glove.6B.50d.txt"

embedding = pretrained_embeddings(file, EMBEDDING_DIM, MAX_NB_WORDS, word_index)
def get_model(MAX_NB_WORDS, EMBEDDING_DIM, MAX_SEQUENCE_LENGTH, embeddings):

    



    input1 = tf.keras.Input(shape=(30,), name='text')

    l = tf.keras.layers.Embedding(MAX_NB_WORDS, EMBEDDING_DIM, input_length=MAX_SEQUENCE_LENGTH, weights=[embeddings], trainable=False)(input1)

    l = tf.keras.layers.CuDNNLSTM(128)(l)

    l = tf.keras.layers.Dense(128, activation='relu')(l)

    l = tf.keras.layers.BatchNormalization()(l)

    

    input2 = tf.keras.Input(shape=(11,), name='meta')

    l2 = tf.keras.layers.BatchNormalization()(input2)

    l2 = tf.keras.layers.Dense(64, activation='relu')(l2)

    l2 = tf.keras.layers.Dense(128, activation='relu')(l2)

    l2 = tf.keras.layers.BatchNormalization()(l2)

    

    outputs = tf.keras.layers.concatenate([l,l2])

    outputs = tf.keras.layers.Dense(64, activation='relu')(outputs)

    outputs = tf.keras.layers.Dropout(0.2)(outputs)

    outputs = tf.keras.layers.Dense(32, activation='relu')(outputs)

    outputs = tf.keras.layers.Dense(2, activation='softmax')(outputs)

    

    

    model = tf.keras.Model(inputs=[input1, input2], outputs=outputs, name='mnist_model')

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])



    return model





from sklearn.utils import class_weight



    

# simple model training. 

# you might want to avoid overfitting by monitoring validation loss and implement early stopping, etc

def train_model(model, X, y,  class_weights):

    model.fit(X, y, epochs=10, batch_size=64, class_weight=class_weights, validation_split=0.2)

    

def predict(model, X):

    y_pred = model.predict(X, batch_size=1024)

    return y_pred
model = get_model(MAX_NB_WORDS, EMBEDDING_DIM, MAX_SEQUENCE_LENGTH, embedding)

model.summary()

tf.keras.utils.plot_model(

    model,

    to_file='model.png',

    show_shapes=True,

    show_layer_names=True,

    rankdir='TB'

)
class_weights = class_weight.compute_class_weight('balanced',

                                                 np.unique(train.label),

                                                 train.label)

class_weights = {0:class_weights[0], 1:class_weights[1]}

print (class_weights)

train_model(model, [X_train, X_train_other], y_train, None)
test_sample_ids = test.id

y_pred = predict(model, [X_test, X_test_other])



# convert predictions to the kaggle format

y_pred_numerical = np.argmax(y_pred, axis = 1) # one-hot to numerical

y_pred_cat = [labels[x] for x in y_pred_numerical] # numerical to string label



# generate the table with the correct IDs for kaggle.

# we get the correct sample ID from the stored array (test_sample_ids)

submission_results = pd.DataFrame({'id':test_sample_ids, 'label':y_pred_cat})

submission_results.to_csv("submission.csv", index=False)