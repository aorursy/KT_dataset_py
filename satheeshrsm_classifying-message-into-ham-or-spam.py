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
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
dataset = pd.read_csv("../input/spam.csv",encoding='latin-1')
##Checking the head

dataset.head()
dataset.describe()
dataset.groupby('v1').describe()
dataset['length'] = dataset['v2'].apply(len)
dataset['length'].plot(bins=50, kind='hist') 
dataset.length.describe()
dataset[dataset['length'] == 910]['v2'].iloc[0]
dataset.hist(column='length', by='v1', bins=50,figsize=(12,4))
dataset.drop(labels = ['Unnamed: 2','Unnamed: 3','Unnamed: 4','length'],axis = 1,inplace = True)
import re

import nltk

from nltk.corpus import stopwords

from nltk.stem.porter import PorterStemmer
portstemmer = PorterStemmer()

corpus = []

for i in range (0,len(dataset)):

    mess = re.sub('[^a-zA-Z]',repl = ' ',string = dataset['v2'][i])

    mess.lower()

    mess = mess.split()

    mess = [portstemmer.stem(word) for word in mess if word not in set(stopwords.words('english'))]

    mess = ' '.join(mess)

    corpus.append(mess)
corpus[1]
len(corpus)
from sklearn.feature_extraction.text import CountVectorizer
countvectorizer = CountVectorizer()
x = countvectorizer.fit_transform(corpus).toarray() #Independent Variable
y = dataset['v1'].values #Dependent Variable
x.shape
from sklearn.model_selection import train_test_split 
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.3)
from sklearn.naive_bayes import MultinomialNB
multinomialnb = MultinomialNB()
multinomialnb.fit(x_train,y_train)
y_pred = multinomialnb.predict(x_test)
y_pred
from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(y_test,y_pred))
confusion_matrix(y_test,y_pred)
from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_pred)