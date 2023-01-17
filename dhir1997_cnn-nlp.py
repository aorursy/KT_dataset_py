import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
import pandas as pd



filepath_dict = {'yelp':   '/kaggle/input/sentiment-labelled-sentences-data-set/sentiment labelled sentences/yelp_labelled.txt',

                 'amazon': '/kaggle/input/sentiment-labelled-sentences-data-set/sentiment labelled sentences/amazon_cells_labelled.txt',

                 'imdb':   '/kaggle/input/sentiment-labelled-sentences-data-set/sentiment labelled sentences/imdb_labelled.txt'}



df_list = []

for source, filepath in filepath_dict.items():

    df = pd.read_csv(filepath, names=['sentence', 'label'], sep='\t')

    df['source'] = source  # Add another column filled with the source name

    df_list.append(df)



df = pd.concat(df_list)

print(df.iloc[0])
from sklearn.model_selection import train_test_split

df_yelp = df[df['source'] == 'yelp']

sentences = df_yelp['sentence'].values

y = df_yelp['label'].values

sentences_train, sentences_test, y_train, y_test = train_test_split(sentences, y, test_size=0.25, random_state=100)
from sklearn.feature_extraction.text import CountVectorizer



vectorizer = CountVectorizer()

vectorizer.fit(sentences_train)



X_train = vectorizer.transform(sentences_train)

X_test  = vectorizer.transform(sentences_test)

X_train
from sklearn.linear_model import LogisticRegression



classifier = LogisticRegression()

classifier.fit(X_train, y_train)

score = classifier.score(X_test, y_test)



print("Accuracy:", score)
for source in df['source'].unique():

    df_source = df[df['source'] == source]

    sentences = df_source['sentence'].values

    y = df_source['label'].values



    sentences_train, sentences_test, y_train, y_test = train_test_split(sentences, y, test_size=0.25, random_state=1000)



    vectorizer = CountVectorizer()

    vectorizer.fit(sentences_train)

    X_train = vectorizer.transform(sentences_train)

    X_test  = vectorizer.transform(sentences_test)



    classifier = LogisticRegression()

    classifier.fit(X_train, y_train)

    score = classifier.score(X_test, y_test)

    print('Accuracy for {} data: {:.4f}'.format(source, score))
from keras.models import Sequential

from keras import layers



input_dim = X_train.shape[1]  # Number of features



model = Sequential()

model.add(layers.Dense(10, input_dim=input_dim, activation='relu'))

model.add(layers.Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()
history = model.fit(X_train, y_train,epochs=25,verbose=False,validation_data=(X_test, y_test),batch_size=40)
loss, accuracy = model.evaluate(X_train, y_train, verbose=False)

print("Training Accuracy: {:.4f}".format(accuracy))

loss, accuracy = model.evaluate(X_test, y_test, verbose=False)

print("Testing Accuracy:  {:.4f}".format(accuracy))
import matplotlib.pyplot as plt

plt.style.use('ggplot')



def plot_history(history):

    acc = history.history['accuracy']

    val_acc = history.history['val_accuracy']

    loss = history.history['loss']

    val_loss = history.history['val_loss']

    x = range(1, len(acc) + 1)



    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)

    plt.plot(x, acc, 'b', label='Training accuracy')

    plt.plot(x, val_acc, 'r', label='Validation accuracy')

    plt.title('Training and validation accuracy')

    plt.legend()

    plt.subplot(1, 2, 2)

    plt.plot(x, loss, 'b', label='Training loss')

    plt.plot(x, val_loss, 'r', label='Validation loss')

    plt.title('Training and validation loss')

    plt.legend()
plot_history(history)
from keras.preprocessing.text import Tokenizer



tokenizer = Tokenizer(num_words=5000)

tokenizer.fit_on_texts(sentences_train)



X_train = tokenizer.texts_to_sequences(sentences_train)

X_test = tokenizer.texts_to_sequences(sentences_test)



vocab_size = len(tokenizer.word_index) + 1  # Adding 1 because of reserved 0 index



print(sentences_train[2])

print(X_train[2])
from keras.preprocessing.sequence import pad_sequences



maxlen = 100



X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)

X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)



print(X_train[0, :])

from keras import layers



embedding_dim = 50



model = Sequential()

model.add(layers.Embedding(input_dim=vocab_size, 

                           output_dim=embedding_dim, 

                           input_length=maxlen))

model.add(layers.Flatten())

model.add(layers.Dense(10, activation='relu'))

model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer='adam',

              loss='binary_crossentropy',

              metrics=['accuracy'])

model.summary()
history = model.fit(X_train, y_train,

                    epochs=15,

                    verbose=False,

                    validation_data=(X_test, y_test),

                    batch_size=14)

loss, accuracy = model.evaluate(X_train, y_train, verbose=False)

print("Training Accuracy: {:.4f}".format(accuracy))

loss, accuracy = model.evaluate(X_test, y_test, verbose=False)

print("Testing Accuracy:  {:.4f}".format(accuracy))

plot_history(history)
from keras.models import Sequential

from keras import layers



embedding_dim = 50



model = Sequential()

model.add(layers.Embedding(input_dim=vocab_size, 

                           output_dim=embedding_dim, 

                           input_length=maxlen))

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

                    batch_size=25)

loss, accuracy = model.evaluate(X_train, y_train, verbose=False)

print("Training Accuracy: {:.4f}".format(accuracy))

loss, accuracy = model.evaluate(X_test, y_test, verbose=False)

print("Testing Accuracy:  {:.4f}".format(accuracy))

plot_history(history)
import numpy as np



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

embedding_matrix = create_embedding_matrix('/kaggle/input/nlpword2vecembeddingspretrained/glove.6B.50d.txt',tokenizer.word_index, embedding_dim)
nonzero_elements = np.count_nonzero(np.count_nonzero(embedding_matrix, axis=1))

nonzero_elements / vocab_size
from keras.models import Sequential

from keras import layers



embedding_dim = 50



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

                    epochs=50,

                    verbose=False,

                    validation_data=(X_test, y_test),

                    batch_size=10)

loss, accuracy = model.evaluate(X_train, y_train, verbose=False)

print("Training Accuracy: {:.4f}".format(accuracy))

loss, accuracy = model.evaluate(X_test, y_test, verbose=False)

print("Testing Accuracy:  {:.4f}".format(accuracy))

plot_history(history)
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

                    epochs=25,

                    verbose=False,

                    validation_data=(X_test, y_test),

                    batch_size=20)

loss, accuracy = model.evaluate(X_train, y_train, verbose=False)

print("Training Accuracy: {:.4f}".format(accuracy))

loss, accuracy = model.evaluate(X_test, y_test, verbose=False)

print("Testing Accuracy:  {:.4f}".format(accuracy))

plot_history(history)