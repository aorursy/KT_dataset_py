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

file = pd.read_csv("../input/spamses/spam.csv")

file
file.head()
list(file.columns)
file= file.drop(["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], axis=1)

file = file.rename(columns={"v1":"label", "v2":"sms"})
file.head()
print (len(file))
file.label.value_counts()
file.describe()
file['length'] =file['sms'].apply(len)

file.head()
import matplotlib.pyplot as plt

import seaborn as sns

file.hist(column='length', by='label', bins=50,figsize=(10,4))
file.loc[:,'label'] =file.label.map({'ham':0, 'spam':1})

print(file.shape)

file.head()
documents = ['Hello, how are you!',

             'Win money, win from home.',

             'Call me now.',

             'Hello, Call hello you tomorrow?']



lower_case_documents = []

lower_case_documents = [d.lower() for d in documents]

print(lower_case_documents)
sans_punctuation_documents = []

import string



for i in lower_case_documents:

    sans_punctuation_documents.append(i.translate(str.maketrans("","", string.punctuation)))

    

sans_punctuation_documents
preprocessed_documents = [[w for w in d.split()] for d in sans_punctuation_documents]

preprocessed_documents
frequency_list = []

import pprint

from collections import Counter



frequency_list = [Counter(d) for d in preprocessed_documents]

pprint.pprint(frequency_list)
from sklearn.feature_extraction.text import CountVectorizer

count_vector = CountVectorizer()
count_vector.fit(documents)

count_vector.get_feature_names()
doc_array = count_vector.transform(documents).toarray()

doc_array
frequency_matrix = pd.DataFrame(doc_array, columns = count_vector.get_feature_names())

frequency_matrix
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(file['sms'], 

                                                    file['label'],test_size=0.20, 

                                                    random_state=1)


count_vector = CountVectorizer()



training_data = count_vector.fit_transform(X_train)



testing_data = count_vector.transform(X_test)
from sklearn.naive_bayes import MultinomialNB

naive_bayes = MultinomialNB()

naive_bayes.fit(training_data,y_train)
predictions = naive_bayes.predict(testing_data)
from sklearn.metrics import accuracy_score, classification_report ,confusion_matrix

print(classification_report(y_test, predictions))
confusion_matrix(y_test, predictions)
print('accuracy score :',accuracy_score(y_test, predictions)*100)