# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
from collections import Counter

import spacy

import re

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix, accuracy_score

from sklearn.model_selection import cross_val_score

from sklearn.tree import DecisionTreeClassifier
df = pd.read_csv('/kaggle/input/newyork-room-rentalads/room-rental-ads.csv')
df.head()
df.shape
df.describe()
df.isnull().sum()
df.dropna(how = 'any', inplace = True)
df['Vague/Not'] = df['Vague/Not'].astype('int64')
df.rename(columns = {"Vague/Not":"Target"},inplace = True)
df.dtypes
df.columns
df.Target = df.Target.astype('category')
len(df[df.duplicated()])
df.drop_duplicates(inplace = True, subset = ['Description'])
df.shape
nlp = spacy.load('en')

def normalize (msg):

    msg = re.sub('[^A-Za-z]+', ' ', msg) #remove special character and intergers

    doc = nlp(msg)

    res=[]

    for token in doc:

        if(token.is_stop or token.is_punct or token.is_currency or token.is_space or len(token.text) <= 2): #Remove Stopwords, Punctuations, Currency and Spaces

            pass

        else:

            res.append(token.lemma_.lower())

    return res
df['Description'] = df['Description'].apply(normalize)
df['Description']
words_collection = Counter([item for sublist in df['Description'] for item in sublist])

freqword = pd.DataFrame(words_collection.most_common(30))

freqword.columns = ['repeated_word','count']
fig, ax = plt.subplots(figsize=(30,25))

sns.barplot(x = 'repeated_word', y = 'count', data = freqword, ax = ax)

plt.show()
df['Description'] = df['Description'].apply(lambda a:' '.join(a))
c = TfidfVectorizer(ngram_range = (1,2)) 

mat = pd.DataFrame(c.fit_transform(df["Description"]).toarray(), columns = c.get_feature_names())

mat
X = mat

y = df["Target"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 0)
classifier = DecisionTreeClassifier()

classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

cm = confusion_matrix(y_test, y_pred)

print(cm)

accuracy_score(y_test, y_pred)