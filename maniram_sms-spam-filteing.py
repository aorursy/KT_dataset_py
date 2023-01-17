
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
print(os.listdir("../input"))

from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier,VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from wordcloud import WordCloud  # library to represent the words in text in pictorial format
data = pd.read_csv('../input/spam.csv',encoding='ISO-8859-1')
data.drop(['Unnamed: 2','Unnamed: 3','Unnamed: 4'],axis=1,inplace=True)
data.head()
data['labels']=data['v1'].map({'ham':0,'spam':1})
data.head()
X=data.v2
Y=data.labels
cv=CountVectorizer(decode_error='ignore')
X=cv.fit_transform(data['v2'])
trainX,testX,trainY,testY=train_test_split(X,Y,test_size=.3,random_state=42)
model=MultinomialNB()
model.fit(trainX,trainY)
print(model.score(testX,testY))

dc=DecisionTreeClassifier()
dc.fit(trainX,trainY)
print(dc.score(testX,testY))
rfc=RandomForestClassifier()
rfc.fit(trainX,trainY)
print(rfc.score(testX,testY))
et=ExtraTreesClassifier()
et.fit(trainX,trainY)
print(et.score(testX,testY))
knc=KNeighborsClassifier()
knc.fit(trainX,trainY)
print(knc.score(testX,testY))
svc=SVC()
svc.fit(trainX,trainY)
print(svc.score(testX,testY))
cv=TfidfVectorizer(decode_error='ignore')
X=cv.fit_transform(data['v2'])
trainX,testX,trainY,testY=train_test_split(X,Y,test_size=.3,random_state=42)
model=MultinomialNB()
model.fit(trainX,trainY)
print(model.score(testX,testY))
svc=SVC()
svc.fit(trainX,trainY)
print(svc.score(testX,testY))

dc=DecisionTreeClassifier()
dc.fit(trainX,trainY)
print(dc.score(testX,testY))

rfc=RandomForestClassifier()
rfc.fit(trainX,trainY)
print(rfc.score(testX,testY))
et=ExtraTreesClassifier()
et.fit(trainX,trainY)
print(et.score(testX,testY))
knc=KNeighborsClassifier()
knc.fit(trainX,trainY)
print(knc.score(testX,testY))

def visualize(label):
    words=''
    for msg in data[data['v1']==label]['v2'] :
        msg=msg.lower()
        words += msg + ' '

    wc=WordCloud(width=600,height=400).generate(words)
    plt.imshow(wc)
    plt.axis('off')
    plt.show()

visualize('spam')
visualize('ham')
data['predictions']=model.predict(X)
data.head()
for msg in data[(data['labels']==1) & (data['predictions']==0)]['v2']:
    print(msg)
    print('\n')

