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
dataset = pd.read_json('/kaggle/input/recipe-ingredients-dataset/train.json')
dataset.head()
import seaborn as sns

import matplotlib.pyplot as plt

plt.figure(figsize = (15,10))

sns.countplot(dataset['cuisine'])

plt.xticks(rotation = 90,fontsize = 12)
words = [' '.join(word) for word in dataset['ingredients']]

dataset['ingredients2'] = words
dataset.head()
dataset.shape
#Removing numbers ,puntuations and other special characters

import re

import nltk

nltk.download('stopwords')

from nltk.corpus import stopwords

from nltk.stem.porter import PorterStemmer

corpus = []

for i in range(0,39774):

    review = re.sub('[^a-zA-Z]',' ',dataset['ingredients2'][i])

    review = review.lower()

    review = review.split()

    ps = PorterStemmer()

    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]

    review = ' '.join(review)

    corpus.append(review)

#Same code below can also be used
dataset['ingredients modified'] = corpus
#This is regex which removes the digits

s='1 1cool co1l coo1'

s=re.sub(r"(\d)", "", s)

print(s)
#This removes all types of brackets

s='hi 1(bye)'

s=re.sub(r'\([^)]*\)', '', s)

print(s)
#This removes the brand names

s='hi 1 Marvel™ hi'

s=re.sub(u'\w*\u2122', '', s)

print(s)


#This removes the stopwords

import re

from nltk.corpus import stopwords

s="I love this phone, its super fast and there's so much new and cool things with jelly bean....but of recently I've seen some bugs."

pattern = re.compile(r'\b(' + r'|'.join(stopwords.words('english')) + r')\b\s*')

s = pattern.sub('', s)

print(s)
#This removes the stopwords and tokenize the sentences

from nltk.tokenize import word_tokenize

from nltk.corpus import stopwords





example_sent = "This is a sample sentence, showing off the stop words filtration."



stop_words = set(stopwords.words('english'))



word_tokens = word_tokenize(example_sent)



filtered_sentence = [w for w in word_tokens if not w in stop_words]



filtered_sentence = []



for w in word_tokens:

    if w not in stop_words:

        filtered_sentence.append(w)



print(word_tokens)

print(filtered_sentence)
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer()

X = vectorizer.fit_transform(dataset['ingredients modified'])



print(X)
Cuisine_unique = dataset['cuisine'].unique()
y = dataset.cuisine
cusine_dict = {Cuisine_unique[i]:i for i in range(0,len(Cuisine_unique))}

inv_cusine_dict = {i:Cuisine_unique[i] for i in range(0,len(Cuisine_unique))}
cusine_dict
inv_cusine_dict
y = y.map(cusine_dict)
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2)
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score

for K in range(10):

    K_value = K+1

    neigh = KNeighborsClassifier(n_neighbors = K_value, weights='uniform', algorithm='auto')

    neigh.fit(X_train, y_train) 

    y_pred = neigh.predict(X_test)

    print("Accuracy is ", accuracy_score(y_test,y_pred)*100,"% for K-Value:",K_value)
neigh = KNeighborsClassifier(n_neighbors = 10, weights='uniform', algorithm='auto')

neigh.fit(X_train, y_train) 

y_pred = neigh.predict(X_test)

print("Accuracy is ", accuracy_score(y_test,y_pred)*100,"%")
from sklearn import svm

lin_clf = svm.LinearSVC(C=1)

lin_clf.fit(X_train, y_train)

y_pred_svc=lin_clf.predict(X_test)

print(accuracy_score(y_test,y_pred)*100)
from sklearn.naive_bayes import MultinomialNB

clf = MultinomialNB().fit(X_train, y_train)

y_pred = clf.predict(X_test)

print(accuracy_score(y_test,y_pred)*100)
from sklearn.linear_model import LogisticRegression

logisticRegr = LogisticRegression(max_iter=  1000)

logisticRegr.fit(X_train, y_train)

y_pred = logisticRegr.predict(X_test)

print(accuracy_score(y_test,y_pred)*100)
res = pd.DataFrame(y_pred_svc,columns =['Predicted'])
df3=pd.DataFrame({'id':y_test.index, 'cuisine':y_test.values})

y_test1=df3['cuisine'].tolist()

y_test1
result=pd.DataFrame({'Actual Cuisine':y_test1, 'Predicted Cuisine':y_pred})

print(result)
result['Actual Cuisine'] =result['Actual Cuisine'].map(inv_cusine_dict) 
result['Predicted Cuisine'] = result['Predicted Cuisine'].map(inv_cusine_dict)
result