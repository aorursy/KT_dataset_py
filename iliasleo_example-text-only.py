# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import tensorflow as tf



# Any results you write to the current directory are saved as output.
train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")

train.head(10)
train.label.value_counts()

train.label.value_counts().plot(kind='bar')
test.head(5)
# Max number of words to be used

MAX_NB_WORDS = 5000

# Max number of words in each complaint (you can also change this)

MAX_SEQUENCE_LENGTH = 160 



# One hot encode the labels too

labels = ['normal', 'sarcastic']



def process_text(train, test):

    # you might want to do some text cleaning

    

    # you might want to stem words

    

    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=MAX_NB_WORDS, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)

    tokenizer.fit_on_texts(train.text) # only fit on train data

    

    # print number of words  found/used

    word_index = tokenizer.word_index

    print('Found %s unique tokens.' % len(word_index))

    

    # tokenizer the train text into words and create the enumeration

    X_train = tokenizer.texts_to_sequences(train.text)

    # pad tweets that are smaller with zero

    X_train = tf.keras.preprocessing.sequence.pad_sequences(X_train, maxlen=MAX_SEQUENCE_LENGTH)

    

    # tokenizer the test text into words and create the enumeration

    X_test = tokenizer.texts_to_sequences(test.text)

    # pad tweets that are smaller with zero

    X_test = tf.keras.preprocessing.sequence.pad_sequences(X_test, maxlen=MAX_SEQUENCE_LENGTH)

    

    # One hot encode the labels too

    y_train = pd.get_dummies(train.label)[labels]

    

    print('Shape of train data tensor:', X_train.shape)

    print('Shape of train label tensor:', y_train.shape)

    print('Shape of test data tensor:', X_test.shape)

    

    return X_train, y_train, X_test, word_index

    



X_train, y_train, X_test, word_index = process_text(train, test)









    
# print the original text of the first tweet

print(train.loc[0, 'text'])

# print how this looks now, after we created the enumeration

print(X_train[0,:])
def get_model(MAX_NB_WORDS, EMBEDDING_DIM, MAX_SEQUENCE_LENGTH):

    model = tf.keras.Sequential()

    # you might want to use embedding (glove, word2vec, etc)

    model.add(tf.keras.layers.Embedding(MAX_NB_WORDS, EMBEDDING_DIM, input_length=MAX_SEQUENCE_LENGTH))

    model.add(tf.keras.layers.CuDNNLSTM(20))

    model.add(tf.keras.layers.Dense(2, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model



    

# simple model training. 

# you might want to avoid overfitting by monitoring validation loss and implement early stopping, etc

def train_model(model, X, y):

    model.fit(X, y, epochs=20, batch_size=256, validation_split=0.2)

    

def predict(model, X):

    y_pred = model.predict(X, batch_size=1024)

    return y_pred
model = get_model(MAX_NB_WORDS, 10, MAX_SEQUENCE_LENGTH)

model.summary()

train_model(model, X_train, y_train)
test_sample_ids = test.id

y_pred = predict(model, X_test)



# convert predictions to the kaggle format

y_pred_numerical = np.argmax(y_pred, axis = 1) # one-hot to numerical

y_pred_cat = [labels[x] for x in y_pred_numerical] # numerical to string label



# generate the table with the correct IDs for kaggle.

# we get the correct sample ID from the stored array (test_sample_ids)

submission_results = pd.DataFrame({'id':test_sample_ids, 'label':y_pred_cat})

submission_results.to_csv("submission.csv", index=False)