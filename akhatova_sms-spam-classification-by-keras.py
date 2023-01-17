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
# IMPORT LIBRARIES



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

plt.style.use('ggplot')



from sklearn.feature_extraction.text import CountVectorizer

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import OneHotEncoder

from sklearn.model_selection import RandomizedSearchCV



from keras.models import Sequential

from keras import layers

from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from keras.wrappers.scikit_learn import KerasClassifier



import os

print(os.listdir("../input"))
df = pd.read_csv('../input/sms-spam-collection-dataset/spam.csv',delimiter=',',encoding='latin-1')

df = df.drop(['Unnamed: 2','Unnamed: 3','Unnamed: 4'],axis=1)

df = df.rename(columns = {'v1':'spam','v2':'text'})

df['spam'].replace(['ham','spam'],[0,1],inplace=True)

df.head()
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(df['text'], df['spam'], test_size=0.1, random_state=42)
from sklearn.feature_extraction.text import CountVectorizer



vectorizer = CountVectorizer()

vectorizer.fit(X_train)



X_train_cv = vectorizer.transform(X_train)

X_test_cv  = vectorizer.transform(X_test)
from sklearn.linear_model import LogisticRegression

from sklearn import metrics



logreg = LogisticRegression()

logreg.fit(X_train_cv, y_train)
y_pred = logreg.predict(X_test_cv)

print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test_cv, y_test)))
from sklearn.metrics import confusion_matrix

confusion_matrix = confusion_matrix(y_test, y_pred)

print(confusion_matrix)
from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))
from sklearn.metrics import roc_auc_score

from sklearn.metrics import roc_curve

logit_roc_auc = roc_auc_score(y_test, y_pred)

fpr, tpr, thresholds = roc_curve(y_test, logreg.predict_proba(X_test_cv)[:,1])

plt.figure()

plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)

plt.plot([0, 1], [0, 1],'r--')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('Receiver operating characteristic')

plt.legend(loc="lower right")

plt.savefig('Log_ROC')

plt.show()
from keras.preprocessing.text import Tokenizer



tokenizer = Tokenizer(num_words=5000)

tokenizer.fit_on_texts(X_train)



X_train_emb = tokenizer.texts_to_sequences(X_train)

X_test_emb = tokenizer.texts_to_sequences(X_test)



vocab_size = len(tokenizer.word_index) + 1  # Adding 1 because of reserved 0 index



print(X_train[0])

print(X_train_emb[0])
from keras.preprocessing.sequence import pad_sequences



maxlen = 100



X_train = pad_sequences(X_train_emb, padding='post', maxlen=maxlen)

X_test = pad_sequences(X_test_emb, padding='post', maxlen=maxlen)



print(X_train[0, :])
import keras

import tensorflow as tf



embedding_dim = 50

model = keras.Sequential()

model.add(keras.layers.Embedding(vocab_size, embedding_dim))

model.add(keras.layers.GlobalMaxPool1D())

model.add(keras.layers.Dense(16, activation=tf.nn.relu))

model.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))

model.compile(optimizer='adam',

              loss='binary_crossentropy',

              metrics=['acc'])

model.summary()
def plot_history(history):

    acc = history.history['acc']

    val_acc = history.history['val_acc']

    loss = history.history['loss']

    val_loss = history.history['val_loss']

    x = range(1, len(acc) + 1)



    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)

    plt.plot(x, acc, 'b', label='Training acc')

    plt.plot(x, val_acc, 'r', label='Validation acc')

    plt.title('Training and validation accuracy')

    plt.legend()

    plt.subplot(1, 2, 2)

    plt.plot(x, loss, 'b', label='Training loss')

    plt.plot(x, val_loss, 'r', label='Validation loss')

    plt.title('Training and validation loss')

    plt.legend()
history = model.fit(X_train, y_train,

                    epochs=20,

                    verbose=False,

                    validation_data=(X_test, y_test),

                    batch_size=16)

loss, accuracy = model.evaluate(X_train, y_train, verbose=False)

print("Training Accuracy: {:.4f}".format(accuracy))

loss, accuracy = model.evaluate(X_test, y_test, verbose=False)

print("Testing Accuracy:  {:.4f}".format(accuracy))

plot_history(history)
def create_embedding_matrix(filepath, word_index, embedding_dim):

    vocab_size = len(word_index) + 1  # Adding again 1 because of reserved 0 index

    embedding_matrix = np.zeros((vocab_size, embedding_dim))



    with open(filepath) as f:

        for line in f:

            word, *vector = line.split()

            if word in word_index:

                idx = word_index[word] 

                embedding_matrix[idx] = np.array(

                    vector, dtype=np.float32)[:embedding_dim]



    return embedding_matrix
embedding_dim = 50

embedding_matrix = create_embedding_matrix('../input/glove6b50dtxt/glove.6B.50d.txt', tokenizer.word_index, embedding_dim)
nonzero_elements = np.count_nonzero(np.count_nonzero(embedding_matrix, axis=1))

nonzero_elements / vocab_size
model = Sequential()

model.add(layers.Embedding(vocab_size, embedding_dim, 

                           weights=[embedding_matrix], 

                           input_length=maxlen, 

                           trainable=False))

model.add(layers.GlobalMaxPool1D())

model.add(layers.Dense(10, activation='relu'))

model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer='adam',

              loss='binary_crossentropy',

              metrics=['accuracy'])

model.summary()
history = model.fit(X_train, y_train,

                    epochs=20,

                    verbose=False,

                    validation_data=(X_test, y_test),

                    batch_size=16)

loss, accuracy = model.evaluate(X_train, y_train, verbose=False)

print("Training Accuracy: {:.4f}".format(accuracy))

loss, accuracy = model.evaluate(X_test, y_test, verbose=False)

print("Testing Accuracy:  {:.4f}".format(accuracy))

plot_history(history)
# add trainable

model = Sequential()

model.add(layers.Embedding(vocab_size, embedding_dim, 

                           weights=[embedding_matrix], 

                           input_length=maxlen, 

                           trainable=True))

model.add(layers.GlobalMaxPool1D())

model.add(layers.Dense(10, activation='relu'))

model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer='adam',

              loss='binary_crossentropy',

              metrics=['accuracy'])

model.summary()
history = model.fit(X_train, y_train,

                    epochs=20,

                    verbose=False,

                    validation_data=(X_test, y_test),

                    batch_size=16)

loss, accuracy = model.evaluate(X_train, y_train, verbose=False)

print("Training Accuracy: {:.4f}".format(accuracy))

loss, accuracy = model.evaluate(X_test, y_test, verbose=False)

print("Testing Accuracy:  {:.4f}".format(accuracy))

plot_history(history)
embedding_dim = 50



model = Sequential()

model.add(layers.Embedding(vocab_size, embedding_dim, input_length=maxlen))

model.add(layers.Conv1D(128, 5, activation='relu'))

model.add(layers.GlobalMaxPooling1D())

model.add(layers.Dense(10, activation='relu'))

model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer='adam',

              loss='binary_crossentropy',

              metrics=['accuracy'])

model.summary()
history = model.fit(X_train, y_train,

                    epochs=20,

                    verbose=False,

                    validation_data=(X_test, y_test),

                    batch_size=16)

loss, accuracy = model.evaluate(X_train, y_train, verbose=False)

print("Training Accuracy: {:.4f}".format(accuracy))

loss, accuracy = model.evaluate(X_test, y_test, verbose=False)

print("Testing Accuracy:  {:.4f}".format(accuracy))

plot_history(history)
def create_model(num_filters, kernel_size, vocab_size, embedding_dim, maxlen):

    model = Sequential()

    model.add(layers.Embedding(vocab_size, embedding_dim, input_length=maxlen))

    model.add(layers.Conv1D(num_filters, kernel_size, activation='relu'))

    model.add(layers.GlobalMaxPooling1D())

    model.add(layers.Dense(10, activation='relu'))

    model.add(layers.Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam',

                  loss='binary_crossentropy',

                  metrics=['accuracy'])

    return model
param_grid = dict(num_filters=[32, 64, 128],

                  kernel_size=[3, 5, 7],

                  vocab_size=[5000], 

                  embedding_dim=[50],

                  maxlen=[100])
from keras.wrappers.scikit_learn import KerasClassifier

from sklearn.model_selection import RandomizedSearchCV



# Main settings

epochs = 20

embedding_dim = 50

maxlen = 100

df = pd.read_csv('../input/sms-spam-collection-dataset/spam.csv',delimiter=',',encoding='latin-1')

df = df.drop(['Unnamed: 2','Unnamed: 3','Unnamed: 4'],axis=1)

df = df.rename(columns = {'v1':'spam','v2':'text'})

df['spam'].replace(['ham','spam'],[0,1],inplace=True)



# Run grid search for each source (yelp, amazon, imdb)



sentences = df['text'].values

y = df['spam'].values



# Train-test split

sentences_train, sentences_test, y_train, y_test = train_test_split(

    sentences, y, test_size=0.20, random_state=1000)



# Tokenize words

tokenizer = Tokenizer(num_words=5000)

tokenizer.fit_on_texts(sentences_train)

X_train = tokenizer.texts_to_sequences(sentences_train)

X_test = tokenizer.texts_to_sequences(sentences_test)



# Adding 1 because of reserved 0 index

vocab_size = len(tokenizer.word_index) + 1



# Pad sequences with zeros

X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)

X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)



# Parameter grid for grid search

param_grid = dict(num_filters=[32, 64, 128],

                  kernel_size=[3, 5, 7],

                  vocab_size=[vocab_size],

                  embedding_dim=[embedding_dim],

                  maxlen=[maxlen])

model = KerasClassifier(build_fn=create_model,

                        epochs=epochs, batch_size=10,

                        verbose=False)

grid = RandomizedSearchCV(estimator=model, param_distributions=param_grid,

                          cv=4, verbose=1, n_iter=5)

grid_result = grid.fit(X_train, y_train)



# Evaluate testing set

test_accuracy = grid.score(X_test, y_test)
print(test_accuracy)

print(grid.best_score_)

print(grid.best_params_)