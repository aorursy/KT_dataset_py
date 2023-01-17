import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from nltk.corpus import stopwords

from nltk.stem.porter import PorterStemmer as psm

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.naive_bayes import GaussianNB

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix
data=pd.read_csv("/kaggle/input/nlp-dataset/R_Review.tsv",sep='\t',quoting=3)
data['Review'][0]

reviews=[]

ps=psm()

import re

for i in range(0,1003):

    r=re.sub('[^a-zA-Z]',' ',data['Review'][i])

    r=r.lower()

    r=r.split()

    r=[ps.stem(word) for  word in r if not word in stopwords.words('english')]

    r=' '.join(r)

    reviews.append(r)

cs=CountVectorizer(max_features=1500)

x=cs.fit_transform(reviews).toarray()

y=data['Liked'].to_numpy()

xtrain,xtest,ytrain,ytest=train_test_split(x,y,train_size=0.8,random_state=0)
cluster=GaussianNB()

cluster.fit(xtrain,ytrain)
ypred=cluster.predict(xtest)

cs=confusion_matrix(ytest,ypred)
cs
