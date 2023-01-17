# importing all necessary libraries to run the code
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk import NaiveBayesClassifier
from sklearn.model_selection import train_test_split
from nltk.stem import PorterStemmer
import re
# using the variable sw to hold all stopwords that are in English
sw = stopwords.words('english')
# reading csv file with the data for analyse
ds = pd.read_csv('../input/googleplaystore_user_reviews.csv')
# checking to see how the data are formatted
ds.head()
# the method info of a dataframe shows us the number of null coluns of our data
ds.info()
# Number of elements before removing the NaN values
print('Size before removing Nan: %s'% len(ds))

# Number of elements after removing the NaN values
ds.dropna(axis=0, inplace=True)
print('Size before removing Nan: %s'% len(ds))
def cleaning_data(data):
    aux_list = []
    flag = False
    for phase_word in data:
        word_list = []
        for word in phase_word.split():
            word = word.lower()
            if flag and not word in sw:
                flag = False
                word_list.append('not_'+word)
                continue
            if re.search('(n\'t)$|(not)|(no)|(never)', word):
                flag = True
                continue
            if not word in sw:
                word = re.sub('[\W_0-9]', ' ', word)
                word_list.append(word)
        aux_list.append(' '.join(word_list))
    return aux_list
X = cleaning_data(ds['Translated_Review'])
y = ds['Sentiment']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
# This CountVectorizer is used to represent the words as a list of values, instead of text
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()

vectorizer.fit(X)
X_train = vectorizer.transform(X_train)
X_test = vectorizer.transform(X_test)

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
le.fit(y)
y_train = le.transform(y_train)
y_test = le.transform(y_test)
from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dense(units=100, activation='relu', input_dim=len(vectorizer.get_feature_names())))
model.add(Dense(units=3, activation='sigmoid'))
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(X_train, y_train, 
          epochs=2, verbose=1)
scores = model.evaluate(X_test, y_test, verbose=1)
print("Accuracy:", scores[1])