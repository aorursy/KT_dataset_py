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

import numpy as np

import pickle

import sys

import os

import io

import re

from sys import path

import numpy as np

import pickle

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.naive_bayes import MultinomialNB

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelBinarizer

import matplotlib.pyplot as plt

from string import punctuation, digits

from IPython.core.display import display, HTML

from nltk.corpus import stopwords

from nltk.corpus import stopwords

from nltk.tokenize import word_tokenize

from nltk.tokenize import RegexpTokenizer



#Amazon Data

input_file = "../input/sentiment-labelled-sentences-data-set/sentiment labelled sentences/sentiment labelled sentences/amazon_cells_labelled.txt"

amazon = pd.read_csv(input_file,delimiter='\t',header=None)

amazon.columns = ['Sentence','Class']

#Yelp Data

input_file = "../input/sentiment-labelled-sentences-data-set/sentiment labelled sentences/sentiment labelled sentences/yelp_labelled.txt"

yelp = pd.read_csv(input_file,delimiter='\t',header=None)

yelp.columns = ['Sentence','Class']

#Imdb Data

input_file = "../input/sentiment-labelled-sentences-data-set/sentiment labelled sentences/sentiment labelled sentences/imdb_labelled.txt"

imdb = pd.read_csv(input_file,delimiter='\t',header=None)

imdb.columns = ['Sentence','Class']

#combine all data sets

data = pd.DataFrame()

data = pd.concat([amazon, yelp, imdb])

data['index'] = data.index



data



#Total Count of Each Category

pd.set_option('display.width', 4000)

pd.set_option('display.max_rows', 1000)

distOfDetails = data.groupby(by='Class', as_index=False).agg({'index': pd.Series.nunique}).sort_values(by='index', ascending=False)

distOfDetails.columns =['Class', 'COUNT']

print(distOfDetails)

#Distribution of All Categories

plt.pie(distOfDetails['COUNT'],autopct='%1.0f%%',shadow=True, startangle=360)

plt.show()



#Text Preprocessing

columns = ['index','Class', 'Sentence']

df_ = pd.DataFrame(columns=columns)

#lower string

data['Sentence'] = data['Sentence'].str.lower()

#remove email adress

data['Sentence'] = data['Sentence'].replace('[a-zA-Z0-9-_.]+@[a-zA-Z0-9-_.]+', '', regex=True)

#remove IP address

data['Sentence'] = data['Sentence'].replace('((25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)(\.|$)){4}', '', regex=True)

#remove punctaitions and special chracters

data['Sentence'] = data['Sentence'].str.replace('[^\w\s]','')

#remove numbers

data['Sentence'] = data['Sentence'].replace('\d', '', regex=True)



#remove stop words

for index, row in data.iterrows():

    word_tokens = word_tokenize(row['Sentence'])

    filtered_sentence = [w for w in word_tokens if not w in stopwords.words('english')]

    df_ = df_.append({"index": row['index'], "Class":  row['Class'],"Sentence": " ".join(filtered_sentence[0:])}, ignore_index=True)

data = df_



X_train, X_test, y_train, y_test = train_test_split(data['Sentence'].values.astype('U'),data['Class'].values.astype('int32'), test_size=0.10, random_state=0)

classes  = data['Class'].unique()



from sklearn.metrics import confusion_matrix

from sklearn.model_selection import cross_val_score

from sklearn.metrics import accuracy_score

from sklearn.neural_network import MLPClassifier

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import SGDClassifier



#grid search result

vectorizer = TfidfVectorizer(analyzer='word',ngram_range=(1,2), max_features=50000,max_df=0.5,use_idf=True, norm='l2') 

counts = vectorizer.fit_transform(X_train)

vocab = vectorizer.vocabulary_

classifier = SGDClassifier(alpha=1e-05,max_iter=50,penalty='elasticnet')

targets = y_train

classifier = classifier.fit(counts, targets)

example_counts = vectorizer.transform(X_test)

predictions = classifier.predict(example_counts)



from sklearn.metrics import precision_score

from sklearn.metrics import recall_score

from sklearn.metrics import classification_report



#Model Evaluation

acc = accuracy_score(y_test, predictions, normalize=True)

hit = precision_score(y_test, predictions, average=None,labels=classes)

capture = recall_score(y_test, predictions, average=None,labels=classes)

print('Model Accuracy:%.2f'%acc)

print(classification_report(y_test, predictions))
