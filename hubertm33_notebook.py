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
from keras.models import Sequential
from keras.layers import Dense
csvFile = pd.read_csv("../input/train.csv")
csvFileTest = pd.read_csv("../input/test.csv")
# print(csvFile.iloc[0])
X, y = csvFile["Reviews"], csvFile["Rating"]
Xtest = csvFileTest["Reviews"]
model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
import re
import nltk
from nltk.stem import PorterStemmer

porter = PorterStemmer()
def stemSentence(sentence):
    # Remove all the special characters
    document = re.sub(r'\W', ' ', str(sentence))

    # remove all single characters
    document = re.sub(r'\s+[a-zA-Z]\s+', ' ', document)

    # Remove single characters from the start
    document = re.sub(r'\^[a-zA-Z]\s+', ' ', document)

    # Substituting multiple spaces with single space
    document = re.sub(r'\s+', ' ', document, flags=re.I)

    # Removing prefixed 'b'
    document = re.sub(r'^b\s+', '', document)

    # Converting to Lowercase
    document = document.lower()

    token_words = nltk.word_tokenize(document)

    # remove all tokens that are not alphabetic
    words = [word for word in token_words if word.isalpha()]

    #stop_words = set(stopwords.words('english'))
    #words = [w for w in words if not w in stop_words]
    #bigrm = nltk.bigrams(tokens)

    # token_words
    stem_sentence = []
    for word in words:
        stem_sentence.append(porter.stem(word))
        stem_sentence.append(" ")
    return "".join(stem_sentence)
X.dropna()
X = X.apply(stemSentence)
Xtest = Xtest.apply(stemSentence)
X
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15)
MAX_SEQ_LENGHT = len(max(X_train, key=len))
print(MAX_SEQ_LENGHT)
from sklearn.feature_extraction.text import CountVectorizer
 
vectorizer = CountVectorizer(binary=True, stop_words=stopwords.words('english'), 
                             lowercase=True, min_df=3, max_df=0.9, max_features=5000)
X_train_onehot = vectorizer.fit_transform(X_train)
X_test_onehot = vectorizer.fit_transform(X_test)
vectorizer = CountVectorizer()
vectorizer.fit(X_train)
X_train_1 = vectorizer.transform(X_train)
X_test_2 = vectorizer.transform(X_test)
Xtest = vectorizer.transform(Xtest)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error

classifier = LogisticRegression()
classifier.fit(X_train_1, y_train)
y_pred = classifier.predict(X_test_2)
score = mean_squared_error(y_test, y_pred)

print("MSE:", score)
y_pred_sub= classifier.predict(Xtest)
df_sub = pd.DataFrame(y_pred_sub)
print(df_sub)
df_sub.to_csv('csv_to_submit.csv', index = False)
