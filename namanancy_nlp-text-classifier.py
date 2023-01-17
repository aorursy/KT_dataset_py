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
sentences = ['John likes ice cream', 'John hates chocolate.']

from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer(min_df=0, lowercase=False)

vectorizer.fit(sentences)

vectorizer.vocabulary_



vectorizer.transform(sentences).toarray()
from sklearn.model_selection import train_test_split



df_yelp = df[df['source'] == 'yelp']



sentences = df_yelp['sentence'].values

y = df_yelp['label'].values



sentences_train, sentences_test, y_train, y_test = train_test_split(sentences, y, test_size=0.25, random_state=1000)
from sklearn.feature_extraction.text import CountVectorizer



vectorizer = CountVectorizer()

vectorizer.fit(sentences_train)



X_train = vectorizer.transform(sentences_train)

X_test  = vectorizer.transform(sentences_test)

X_train



from sklearn.linear_model import LogisticRegression



classifier = LogisticRegression(solver='lbfgs')

classifier.fit(X_train, y_train)

score = classifier.score(X_test, y_test)



print("Accuracy:", score)

for source in df['source'].unique():

    df_source = df[df['source'] == source]

    sentences = df_source['sentence'].values

    y = df_source['label'].values



    sentences_train, sentences_test, y_train, y_test = train_test_split(

        sentences, y, test_size=0.25, random_state=1000)



    vectorizer = CountVectorizer()

    vectorizer.fit(sentences_train)

    X_train = vectorizer.transform(sentences_train)

    X_test  = vectorizer.transform(sentences_test)



    classifier = LogisticRegression(solver='lbfgs')

    classifier.fit(X_train, y_train)

    score = classifier.score(X_test, y_test)

    print('Accuracy for {} data: {:.4f}'.format(source, score))
from keras.models import Sequential

from keras import layers



input_dim = X_train.shape[1]  # Number of features



model = Sequential()

model.add(layers.Dense(10, input_dim=input_dim, activation='relu'))

model.add(layers.Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', 

              optimizer='adam', 

              metrics=['accuracy'])

model.summary()
history = model.fit(X_train, y_train,

                    epochs=100,

                    verbose=False,

                    validation_data=(X_test, y_test),

                    batch_size=10)
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

    plt.plot(x, acc, 'b', label='Training acc')

    plt.plot(x, val_acc, 'r', label='Validation acc')

    plt.title('Training and validation accuracy')

    plt.legend()

    plt.subplot(1, 2, 2)

    plt.plot(x, loss, 'b', label='Training loss')

    plt.plot(x, val_loss, 'r', label='Validation loss')

    plt.title('Training and validation loss')

    plt.legend()

plot_history(history)