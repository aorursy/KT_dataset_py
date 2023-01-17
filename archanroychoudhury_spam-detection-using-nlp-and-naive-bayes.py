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
# Impotant Imports

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style('whitegrid')

%matplotlib inline
data = pd.read_csv("../input/spam.csv", encoding='latin-1')
data.head()
data.columns = ['label', 'message', 'line1', 'line2', 'line3']
data.info()
data.describe()
data.drop(['line1','line2', 'line3'], axis=1, inplace=True)
data.head()
data.groupby('label').describe()
data['length'] = data['message'].apply(len)

data.head()
data['length'].plot(bins=50, kind='hist', cmap='coolwarm')

plt.show()
data.length.describe()
data[data['length'] == 910]['message'].iloc[0]
data.hist(column='length', by='label', bins=50,figsize=(12,4))

plt.show()
import nltk

import string

from nltk.corpus import stopwords

stopwords.words('english')[0:10]
def remove_stopword(mess):

    # Checking characters if they are in punctuation

    message = [char for char in mess if char not in string.punctuation]



    # Joining the characters 

    message = ''.join(message)

    

    # Removing any stopwords

    return [word for word in message.split() if word.lower() not in stopwords.words('english')]
data['message'].head(5).apply(remove_stopword)
data.head()
from sklearn.model_selection import train_test_split



msg_train, msg_test, label_train, label_test = train_test_split(data['message'], data['label'], test_size=0.3, random_state=101)



print(len(msg_train), len(msg_test), len(msg_train) + len(msg_test))
from sklearn.pipeline import Pipeline

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.naive_bayes import MultinomialNB



pipeline = Pipeline([

    ('vector', CountVectorizer(analyzer=remove_stopword)),  # strings to tokenized vectors

    ('tfidf', TfidfTransformer()),  # integer counts to weighted TF-IDF scores

    ('classifier', MultinomialNB()),  # train on TF-IDF vectors with Naive Bayes classifier

])
pipeline.fit(msg_train,label_train)
predictions = pipeline.predict(msg_test)
from sklearn.metrics import classification_report,confusion_matrix



print(confusion_matrix(label_test,predictions))

print("\n")

print(classification_report(label_test,predictions))
from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split

accuracy = accuracy_score(label_test,predictions)

print("Accuracy: %.2f%%" % (accuracy * 100.0))