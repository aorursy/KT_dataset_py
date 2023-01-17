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
train=pd.read_csv('/kaggle/input/fake-news/train.csv')

test=pd.read_csv('/kaggle/input/fake-news/test.csv')

submit=pd.read_csv('/kaggle/input/fake-news/submit.csv')
train.head()
X=train.drop('label',axis=1)
Y=train.label
Y.head()
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer,HashingVectorizer
train=train.dropna()

mes=train.copy()
mes.reset_index(inplace=True)

mes.head()
from nltk.corpus import stopwords

from nltk.stem import PorterStemmer

import re 
ps=PorterStemmer()

corpus=[]

for i in range(0,len(mes)):

    r=re.sub('[^a-zA-Z]',' ',mes['title'][i])

    r=r.lower()

    r=r.split()

    r=[ps.stem(word) for word in r if not word in stopwords.words('english')]

    r=' '.join(r)

    corpus.append(r)
corpus
cv=CountVectorizer(max_features=6000,ngram_range=(1,3))

X=cv.fit_transform(corpus).toarray()
X.shape
Y=mes['label']
from sklearn.model_selection import train_test_split

xtrain,xtest,ytrain,ytest=train_test_split(X,Y,test_size=0.33,random_state=113)
#cv.get_feature_names()[:20]
from sklearn.naive_bayes import MultinomialNB

ml=MultinomialNB()
ml.fit(xtrain,ytrain)

pred=ml.predict(xtest)
pred