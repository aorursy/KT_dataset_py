import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
df = pd.read_csv('../input/spamtest/spam.csv')
df
#Plotting a bar graph to see the number of spam and not spam emails.
import matplotlib.pyplot as plt
spam = (df.loc[df['Label'] == 'spam'])['Label'].count()
notspam = ((df.loc[df['Label'] == 'ham'])['Label'].count())
labels = ['Spam', 'Not Spam']
values = [spam, notspam]
print(values)
plt.bar(labels,values)
plt.show()

import string
import nltk
from nltk.corpus import stopwords
def text_preprocess(text):
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = [word for word in text.split() if word.lower() not in  stopwords.words('english')]
    return " ".join(text)
email=[]
for text in df['EmailText']:
    email.append(text_preprocess(text))
type(email) # contains the list of pre processed emails
df['EmailText'] = email
df
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()
x = vectorizer.fit_transform(df['EmailText'])
X=np.array(x.toarray())
np.shape(X)
y=[]
for pred in df['Label']:
    if(pred == 'spam'):
        y.append(1)
    else:
        y.append(0)
a= np.ones((np.shape(X)[0],1))
X = np.append(a,X,axis=1)
np.shape(X)
#Splitting the test data
from sklearn.model_selection import train_test_split
(TrainX,ValuateX,Trainy,Valuatey) = train_test_split(X,y,random_state=1)
print(np.shape(TrainX))
print(np.shape(Trainy))
print(np.shape(ValuateX))
print(np.shape(Valuatey))
import math
def hypothesis(theta,X):
    h = -1*(np.matmul(X,theta))
    h = 1/(1+np.exp(h))
    return h
def cost(theta,X,y,m):
    t1 = np.ones(np.shape(y)[0])
    h = hypothesis(theta,X)
    t2 = np.ones(np.shape(h)[0])
    c = np.sum((y*np.log(h)) +((t1+y)*np.log(t2-h) ))/m
    return c
def gradientDescent(theta,X,y,m,alpha,it):
    iteration = []
    c = []
    for i in range(0,it):
       # d = alpha*((hypothesis(theta,X)-y)*X)
        d = hypothesis(theta,X)-y
        d = np.sum((X*d[0])*alpha,axis=0)
       # print(np.shape(d))
        theta = theta - d
        temp = cost(theta,X,y,m)
        #print(temp)
        c.append(-1*cost(theta,X,y,m))
        iteration.append(i)
    return(theta,iteration,c)
np.shape(TrainX)[1] #no of features
theta = np.zeros(np.shape(TrainX)[1])
np.shape(TrainX)
(theta,i,c) = gradientDescent(theta,TrainX,Trainy,np.shape(TrainX[0]),0.000001,2000)
%matplotlib inline
from matplotlib import pyplot as plt
plt.plot(i,c)
plt.xlabel('Iterations ->')
plt.ylabel('Cost function ->')
theta
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

Spam_model = LogisticRegression(solver='liblinear', penalty='l1')
Spam_model.fit(TrainX, Trainy)
pred = Spam_model.predict(ValuateX)
accuracy_score(Valuatey,pred)
