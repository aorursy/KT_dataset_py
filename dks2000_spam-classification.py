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
import seaborn as sns

import matplotlib.pyplot as plt

import keras

from sklearn.model_selection import train_test_split

import operator

from keras.models import Sequential

from keras.layers import Dense

from keras.layers import Flatten

from keras.layers.embeddings import Embedding

from keras.preprocessing import sequence

import nltk as nlp

import re

from nltk.tokenize import word_tokenize 

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.feature_extraction.text import CountVectorizer 
df = pd.read_csv("../input/spam-text-message-classification/SPAM text message 20170820 - Data.csv")

print (df.head())																			#Prints first 5 rows

print ("--------------------------------------------------------------------------")

print (df.shape)																			#Dimensions of the data

print (df.columns.values)																	#Names of columns in the data

print (df.info())																			#Describes data types of each column

print ("--------------------------------------------------------------------------")

print (df.describe())																		#Gives statitics of the data

print ("--------------------------------------------------------------------------")

df["Category"] = [1 if each == "spam" else 0 for each in df["Category"]]

print (df.head())

print ("--------------------------------------------------------------------------")

print (":::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::")
X = df['Message'].values

y = df['Category'].values



X_words = np.unique(np.hstack(X))

y_words = np.unique(y)



print (X_words)
y_word_count = {}



for i in y_words:

	y_word_count[i] = 0

for j in y:

	y_word_count[j] += 1

	

print (y_word_count)

plt_ = sns.barplot(list(y_word_count.keys()), list(y_word_count.values()))

plt_.set_xticklabels(plt_.get_xticklabels(), rotation=90)

plt.show()

print ("--------------------------------------------------------------------------")
X_word_count = {}



for i in X_words:

	X_word_count[i] = 0

for j in X:

	X_word_count[j] += 1



X_word_count = sorted(X_word_count.items(), key=operator.itemgetter(0))

wordlength = [len(x) for x in X]

print ("Mean Word Length",np.mean(wordlength))

print ("Standard Deviation of Word Length",np.std(wordlength))

print ("--------------------------------------------------------------------------")
plt.boxplot(wordlength)

plt.show()

print (":::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::")

print ("--------------------------------------------------------------------------")
def remove_punctuation(text):

    '''a function for removing punctuation'''

    import string

    # replacing the punctuations with no space, 

    # which in effect deletes the punctuation marks 

    translator = str.maketrans('', '', string.punctuation)

    # return the text stripped of punctuation marks

    return text.translate(translator)



def lowerwords(text):

	text = re.sub("[^a-zA-Z]"," ",text) # Excluding Numbers

	text = [word.lower() for word in text.split()]

    # joining the list of words with space separator

	return " ".join(text)

        

df['Message'] = df['Message'].apply(remove_punctuation)

df['Message'] = df['Message'].apply(lowerwords)



print (df.head(10))

print ("--------------------------------------------------------------------------")

description_list = []

for description in df["Message"]:

    description = nlp.word_tokenize(description)

    #description = [ word for word in description if not word in set(stopwords.words("english"))]

    lemma = nlp.WordNetLemmatizer()

    description = [lemma.lemmatize(word) for word in description]

    description = " ".join(description)

    description_list.append(description) # we hide all word one section

   

top_words = max_features = 10000 # We use the most common word

count_vectorizer = CountVectorizer(max_features = max_features, stop_words = "english")

sparce_matrix = count_vectorizer.fit_transform(description_list).toarray()

# print("the most using {} words: {}".format(max_features,count_vectorizer.get_feature_names()))

print (":::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::")

X = sparce_matrix

y = df['Category'].values



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)



print (X_train)

    

print ("Shape of Training Set is",X_train.shape)

print ("Shape of Test Set is",X_test.shape)

print ("--------------------------------------------------------------------------")



print ("Classes",np.unique(y)) 

print ("No.of unique words",len(np.unique(np.hstack(X))))

print ("--------------------------------------------------------------------------")
max_words = 1000

X_train = sequence.pad_sequences(X_train, maxlen=max_words)

X_test = sequence.pad_sequences(X_test, maxlen=max_words)
model = Sequential()

model.add(Embedding(top_words, 32, input_length=max_words))

model.add(Flatten())

model.add(Dense(250, activation='relu'))

model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=128, verbose=2)

scores = model.evaluate(X_test, y_test, verbose=0)

print("Accuracy: %.2f%%" % (scores[1]*100))