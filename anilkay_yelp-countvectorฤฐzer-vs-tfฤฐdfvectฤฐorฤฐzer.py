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
yelp=pd.read_csv("../input/yelp.csv")
yelp
stars=yelp["stars"]
texts=yelp["text"]
texts
scoredata=stars.map({0: 0, 1:0,2:0,3:1,4:1,5:1}) 
scoredata
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer=TfidfVectorizer()

textvector=vectorizer.fit_transform(texts)
textvector
import math
k=math.sqrt(len(texts))*0.5 #Rule of thumb for k-nearast-neighbour algorithm
k=round(k)
k 
from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=k)
from sklearn.model_selection import train_test_split
x_train, x_test,y_train,y_test = train_test_split(textvector,scoredata,test_size=0.20, random_state=0)
knn.fit(x_train,y_train)
ypred=knn.predict(x_test)
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,ypred)
cm
from sklearn.feature_extraction.text import CountVectorizer
covectorize=CountVectorizer()
countvector=covectorize.fit_transform(texts)
countvector
x_train, x_test,y_train,y_test = train_test_split(countvector,scoredata,test_size=0.20, random_state=0)
knn2=KNeighborsClassifier(n_neighbors=k)
knn2.fit(x_train,y_train)
ypred=knn2.predict(x_test)
cm2=confusion_matrix(y_test,ypred)
cm2
import re
text2=[]
for tex in texts:
    tek=re.sub("[^a-zA-Z]"," ",tex)
    text2.append(tek)
text2[4]    

text2[0]

print("Hello world")
text3=[]
for tex in text2:
    tex=tex.lower()
    text3.append(tex)
text3[0]    
vectorizer=TfidfVectorizer()
textvector=vectorizer.fit_transform(text3)
textvector
x_train, x_test,y_train,y_test = train_test_split(textvector,scoredata,test_size=0.20, random_state=0)
knn3=KNeighborsClassifier(n_neighbors=k)
knn3.fit(x_train,y_train)
ypred=knn3.predict(x_test)
cm3=confusion_matrix(y_test,ypred)
cm3
cm2
cm
text3[4]
from nltk.stem.porter import *
from nltk.corpus import stopwords
derlem=[]
ps=PorterStemmer()
for yorum in text3:
    yorum= yorum.split()
    yorlist=[ps.stem(kelime) for kelime in yorum if not kelime in stopwords.words('english')]
    yorumson=' '.join(yorlist)
    derlem.append(yorumson)
    len(derlem)
    #if not kelime in set(stopwords.words('english')
len(derlem)
derlem[4]
textvectorwithfull=vectorizer.fit_transform(derlem) #All Preporecessing is done
textvectorwithfull
x_train, x_test,y_train,y_test = train_test_split(textvectorwithfull,scoredata,test_size=0.20, random_state=0)
knn4=KNeighborsClassifier(n_neighbors=k)
knn4.fit(x_train,y_train)
ypred=knn4.predict(x_test)
cm4=confusion_matrix(y_test,ypred)
cm4





















