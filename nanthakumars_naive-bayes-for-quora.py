# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import nltk

from sklearn.model_selection import train_test_split

from sklearn.naive_bayes import MultinomialNB

from sklearn.feature_extraction.text import CountVectorizer

from wordcloud import WordCloud

import matplotlib.pyplot as plt

from sklearn.metrics import f1_score,accuracy_score



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data= pd.read_csv("/kaggle/input/quora-insincere-questions-classification/train.csv")

data.shape
data.head()
data['target'].value_counts()
data['target'].value_counts(normalize=True)*100
# For Understanding purpose

docs = data["question_text"].str.lower()

# Corpus is called as collection of docs

# Document is collection of terms

# Terms are collection of words

# Corpus ---> Documents -->Terms --> Words

docs = docs.str.replace("[^a-z\s#@]", "")  #Retain alphabets, spaces, hastags, @ & spaces and remove everything else

docs.head()
#nltk.download("stopwords")

docs= data[data['target']==0]["question_text"].str.lower()

wc= WordCloud().generate(" ".join(docs))

plt.figure(figsize=(14,4))

plt.imshow(wc)

plt.axis(False)
docs= data[data['target']==1]['question_text'].str.lower()

wc= WordCloud().generate(" ".join(docs))

plt.figure(figsize=(14,4))

plt.imshow(wc)

plt.axis(False)
stopwords= nltk.corpus.stopwords.words("english")

stopwords.extend([""]) #Extend custom stopwords

stemmer= nltk.stem.PorterStemmer() # identifying root form of the word

import re



def clean_doc(doc):

    doc= doc.lower()

    doc= re.sub('[^a-z\s]',"",doc)

    words = doc.split(" ")

    words_imp= [stemmer.stem(word) for word in words if word not in stopwords]

    doc_cleaned= " ".join(words_imp)

    return doc_cleaned
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer

#df_dtm= pd.DataFrame(dtm.toarray(), columns=vectorizer.get_feature_names()) --> Memory error

## Rows ---> Documents

## Columns ---> tERMS

## Values --> Frequency of each term in a document

X_train,X_test,y_train,y_test= train_test_split(data["question_text"].apply(clean_doc)

                                                ,data["target"],test_size=0.8,random_state=1)
vectorizer = CountVectorizer(min_df=10).fit(X_train)

dtm_train= vectorizer.transform(X_train)

dtm_validate=vectorizer.transform(X_test)
model=MultinomialNB().fit(dtm_train,y_train)
y_pred= model.predict(dtm_validate)

from sklearn.metrics import f1_score,accuracy_score

print(accuracy_score(y_test,y_pred))

print(f1_score(y_test,y_pred))