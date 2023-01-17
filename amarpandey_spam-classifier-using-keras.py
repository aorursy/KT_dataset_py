# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import keras

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split
data = pd.read_csv("../input/spam-text-message-classification/SPAM text message 20170820 - Data.csv")

data.head()
x = data['Message']

y = data['Category'].values



y = np.where(y == 'ham', 0, 1)
num_classes = len(np.unique(y))



print('# of examples: {}'.format(len(x)))

print('# of classes: {}'.format(num_classes))
import string

import nltk



from nltk.corpus import stopwords

from nltk.tokenize import RegexpTokenizer

from nltk.stem import WordNetLemmatizer



x.head(10)
x = x.apply(lambda q : RegexpTokenizer('\w+').tokenize(q))

x.head(10)
def remove_punctuation(text):

    return [c.lower() for c in text if c.lower() not in string.punctuation]



x = x.apply(lambda q : remove_punctuation(q))



print(x.head(10))
def remove_stopwords(tokens):

    return [w for w in tokens if w not in stopwords.words('english')]



x = x.apply(lambda q : remove_stopwords(q))



print(x.head(10))
lemmatizer = WordNetLemmatizer()



def word_lemmatizer(text):

    lem_text = [lemmatizer.lemmatize(w) for w in text]

    return lem_text



x = x.apply(lambda q : word_lemmatizer(q))



print(x.head(10))
# converting df to np array

x = x.values
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)
data = pd.read_csv("../input/spam-text-message-classification/SPAM text message 20170820 - Data.csv")

data.head()
x = data['Message'].values
from keras.preprocessing.text import Tokenizer



tokenizer = Tokenizer(num_words=20000)



tokenizer.fit_on_texts(x)



x = tokenizer.texts_to_matrix(x, mode='tfidf')
print(x[1])

print(y[1])
from keras.models import Sequential

from keras.layers import Dense, Activation, Dropout



model = Sequential()



model.add(Dense(512, input_dim=20000))

model.add(Activation('relu'))

model.add(Dropout(0.3))



model.add(Dense(1))

model.add(Activation('sigmoid'))



model.summary()
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
_ = model.fit(x, y, batch_size=32, epochs=1, verbose=1, validation_split=0.2)