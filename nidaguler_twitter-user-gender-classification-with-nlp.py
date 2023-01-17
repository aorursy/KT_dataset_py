# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import re

import nltk

import nltk as nlp

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import warnings

warnings.filterwarnings("ignore")

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df=pd.read_csv(r"../input/twitter-user-gender-classification/gender-classifier-DFE-791531.csv",encoding="latin1")

df=pd.concat([df.gender,df.description],axis=1)
df.dropna(axis=0,inplace=True)
df.gender=[1 if each =="female" else 0 for each in df.gender]
df.head()
description_list=[]

for description in df.description:

    description=re.sub("[^a-zA-Z]"," ",description)

    description=description.lower()

    description=nltk.word_tokenize(description)

    #description=[word for word in description if not word in set(stopwords.words("english"))]

    lemma  = nlp.WordNetLemmatizer()

    description=[lemma.lemmatize(word) for word in description]

    description=" ".join(description)

    description_list.append(description)
#bag of words

from sklearn.feature_extraction.text import CountVectorizer

max_features =5000

count_vectorizer =CountVectorizer(max_features=max_features,stop_words="english")

sparce_matrix = count_vectorizer.fit_transform(description_list).toarray()

print("The 5000 most commonly used {} words: {} ".format(max_features,count_vectorizer.get_feature_names()))
y=df.iloc[:,0].values

x=sparce_matrix
#train test split

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.1,random_state=10)
#naive bayes

from sklearn.naive_bayes import GaussianNB

nb=GaussianNB()

nb.fit(x_train,y_train)
#prediction

y_pred=nb.predict(x_test)

print("accuracy: ",nb.score(y_pred.reshape(-1,1),y_test))