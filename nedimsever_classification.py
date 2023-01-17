# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



data=pd.read_csv("../input/questions.csv")
#Imports



import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

data.head(10)
cols=data.drop(['id','qid1','qid2'],axis=1) #i will not use id,qid1 and qid2 
cols.info()
cols.isnull().sum()
#Empty rows are few. So i decided to drop empty ones.
quora=cols.dropna()
quora.isnull().sum()
quora['length q1'] = quora['question1'].apply(len)

quora['length q2']= quora['question2'].apply(len) # i will use this informations in EDA part.
quora.head()
quora.describe()
plt.figure(figsize=(12,6))

sns.set_style('whitegrid')

sns.distplot(quora['length q1'],color='red',bins=50)
plt.figure(figsize=(12,6))

sns.distplot(quora['length q2'],color='green',bins=50)
plt.figure(figsize=(12,6))

g = sns.FacetGrid(quora,col='is_duplicate')

g.map(plt.scatter,'length q1','length q2')
#It is easy so that there is a positive correlation between length q1 and length q2 when they are duplicate. It is hard to say 

#for when they are not. 
sns.jointplot(x='length q1',y='length q2',data=quora,kind='scatter',color='purple')
#it is hard to say strong correlation between q1 and q2.(0.48) 
#lets check the outliers
plt.figure(figsize=(12,6))

sns.boxplot(x='is_duplicate',y='length q1',data=quora,palette='rainbow')
plt.figure(figsize=(12,6))

sns.boxplot(x='is_duplicate',y='length q2',data=quora,palette='rainbow')
sns.countplot(x='is_duplicate',data=quora,palette='viridis')
sns.heatmap(quora.corr(),cmap='coolwarm',annot=True)
import string

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
X=quora['question1']+quora['question2']

y=quora['is_duplicate']
cv=CountVectorizer(stop_words='english')
X=cv.fit_transform(X)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.3,random_state=101)
#Model Comparison
#Due to high production costs of ensemble, KNN and SVC; I will compare LogisticRegression and MultinımialNB.
from sklearn.linear_model import LogisticRegression

from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
classifiers=[['Logistic Regression :',LogisticRegression()],

       ['MultinomialNB:',MultinomialNB()]]

cla_pred=[]

for name,model in classifiers:

    model=model

    model.fit(X_train,y_train)

    predictions = model.predict(X_test)

    cla_pred.append(accuracy_score(y_test,predictions))

    print(name,accuracy_score(y_test,predictions))
y_axis=['Logistic Regression','MultinonialNB']

x_axis=cla_pred
sns.barplot(x=x_axis,y=y_axis)

plt.xlabel('Accuracy')
#Lets add TfidTransformer
from sklearn.feature_extraction.text import  TfidfTransformer
from sklearn.pipeline import Pipeline
pipeline = Pipeline([

    ('bow', CountVectorizer(stop_words = 'english')),  # strings to token integer counts

    ('tfidf', TfidfTransformer()),  # integer counts to weighted TF-IDF scores

    ('classifier', LogisticRegression()),  # train on TF-IDF vectors w/ Naive Bayes classifier

])
X=quora['question1']+quora['question2']

y=quora['is_duplicate']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
pipeline.fit(X_train,y_train)
predictions2=pipeline.predict(X_test)
print(accuracy_score(y_test,predictions2))

print(classification_report(y_test,predictions2))

print(confusion_matrix(y_test,predictions2))
#Accuracy score did not change much. 