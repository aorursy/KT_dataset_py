# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
dataframe_1 = pd.read_csv("../input/deepnlp/Sheet_1.csv")

dataframe_1.head()
mapping_flag = {'flagged': 1, 'not_flagged': 0}



dataframe_1 = dataframe_1.replace({'class': mapping_flag})



response = dataframe_1['response_text']

flag = dataframe_1['class']



print(flag)

print(response)
#vectorize the sentences

from sklearn.feature_extraction.text import CountVectorizer



vectorizer = CountVectorizer(min_df=0, lowercase=False)

vectorizer.fit(response)

vectorizer.vocabulary_
#I want to use now basic NLP tools

import nltk

print(nltk.__version__)

from nltk.corpus import wordnet



word_ = wordnet.synsets("seem")[0]

word_ = np.asarray(word_)



print(word_)



for i in response:

    word = 'Good'.lower()

    if word in i:

        print('Seems a good review')
positive_words = ['awesome','good','nice','fun']

negative_words = ['awful','lame','horrible','bad'] 



from nltk.corpus import stopwords

from nltk import word_tokenize



stop_words = set(stopwords.words('english'))

print(response[0])



response_words = word_tokenize(response[0])

response_words = [w for w in response_words if not w in stop_words]



print(response_words)



for i in response:

    i = i.lower()

    i = i.replace('!','')

    i = i.replace('?','')    
from textblob.classifiers import NaiveBayesClassifier

from textblob import TextBlob



train = [('I love this sandwich.', 'pos'),

         ('This is an amazing place!', 'pos'),

         ('I feel very good about these beers.', 'pos'),

         ('This is my best work.', 'pos'),

         ("What an awesome view", 'pos'),

         ('I do not like this restaurant', 'neg'),

         ('I am tired of this stuff.', 'neg'),

         ("I can't deal with this", 'neg'),

         ('He is my sworn enemy!', 'neg'),

         ('My boss is horrible.', 'neg') ]



model = NaiveBayesClassifier(train)



for i in range(0,len(response)):

    print(model.classify(response[i]))
import keras 

from keras.preprocessing.text import Tokenizer



from sklearn.model_selection import train_test_split



sentences_train, sentences_test, y_train, y_test = train_test_split(response, flag, test_size=0.25, random_state=1000)



tokenizer = Tokenizer(num_words=5000)

tokenizer.fit_on_texts(sentences_train)



X_train = tokenizer.texts_to_sequences(sentences_train)

X_test = tokenizer.texts_to_sequences(sentences_test)



vocab_size = len(tokenizer.word_index) + 1



from keras.preprocessing.sequence import pad_sequences



maxlen = 100



X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)

X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)
#let's try to build a DNN for text processing

#X_train = vectorizer.transform(sentences_train)

#X_test  = vectorizer.transform(sentences_test)



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
import matplotlib.pyplot as plt

plt.style.use('ggplot')



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

                    batch_size=10)

loss, accuracy = model.evaluate(X_train, y_train, verbose=False)

print("Training Accuracy: {:.4f}".format(accuracy))

loss, accuracy = model.evaluate(X_test, y_test, verbose=False)

print("Testing Accuracy:  {:.4f}".format(accuracy))

print(accuracy)

plot_history(history)
#optimization in the hyperparameter space (credits: https://realpython.com/python-keras-text-classification/)

from keras.wrappers.scikit_learn import KerasClassifier

from sklearn.model_selection import RandomizedSearchCV



# Main settings

epochs = 20

embedding_dim = 50

maxlen = 100

output_file = 'data/output.txt'



def create_model_embedding(num_filters, kernel_size, vocab_size, embedding_dim, maxlen):

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



def create_model_LSTM(num_filters, kernel_size, vocab_size, embedding_dim, maxlen):

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



# Run grid search for each source (yelp, amazon, imdb)

for source, frame in dataframe_1.groupby('class'):

    print('Running grid search for data set :', source)



    # Train-test split

    sentences_train, sentences_test, y_train, y_test = train_test_split(

        response, flag, test_size=0.25, random_state=1000)



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



    # Save and evaluate results

    #prompt = input(f'finished {source}; write to file and proceed? [y/n]')

    #if prompt.lower() not in {'y', 'true', 'yes'}:

    #    break

    #with open(output_file, 'a') as f:

    s = ('Running {} data set\nBest Accuracy : '

         '{:.4f}\n{}\nTest Accuracy : {:.4f}\n\n')

    output_string = s.format(

                source,

            grid_result.best_score_,

            grid_result.best_params_,

            test_accuracy)

    print(output_string)

    #f.write(output_string)