# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

spams=pd.read_table("../input/SMSSpamCollection.txt")

spams.head(5)
print(type(spams))
spams.shape
print("list of columns: ",spams.columns,"\n")

print("Type of columns:\n ",spams.dtypes,"\n")

print("Description : \n" ,spams.describe())
pd.crosstab(index=spams["classe"],columns="count")
from sklearn.model_selection import train_test_split

spamsTrain,spamsTest=train_test_split(spams,train_size=3572,random_state=1,stratify=spams['classe'])
#Import the countVectorizer tool

from sklearn.feature_extraction.text import CountVectorizer 

#instantiation of the object - binary weighting

parseur=CountVectorizer(binary=True)

#Create the document term matrix

XTrain=parseur.fit_transform(spamsTrain['message'])

print("number of tokens : " ,len(parseur.get_feature_names()))

print("list of tokens : " ,parseur.get_feature_names())
#Transform the spam matrix into numpy matrix

mdtTrain=XTrain.toarray()

print(type(mdtTrain))

print("size of matrix : ",mdtTrain.shape)
#Frequency of terms

freq_mots=np.sum(mdtTrain,axis=0)

print(freq_mots)
index = np.argsort(freq_mots)

print(index)

#print the terms and their frequency

imp={'terme':np.asarray(parseur.get_feature_names())[index],'freq':freq_mots[index]}

print(pd.DataFrame(imp))
#import the class LogistiRegression

from sklearn.linear_model import LogisticRegression

#instatiate the object

lr=LogisticRegression()

#perform the training process

lr.fit(mdtTrain,spamsTrain["classe"])
#size of coefficients matrix 

print(lr.coef_.shape)

#intercept of the model 

print(lr.intercept_)
#create the document term matrix 

mdtTest = parseur.transform(spamsTest['message'])

#size of the matrix 

print(mdtTest.shape)
predTest = lr.predict(mdtTest) 

predTest
#recall

print(metrics.recall_score(spamsTest['classe'],predTest,pos_label='spam'))
#precision

print(metrics.precision_score(spamsTest['classe'],predTest,pos_label='spam'))

#F1-Score

print(metrics.f1_score(spamsTest['classe'],predTest,pos_label='spam'))
#accuracy rate

print(metrics.accuracy_score(spamsTest['classe'],predTest))
#rebuild the parser with new options : stop_words='english' and min_df = 10

parseurBis=CountVectorizer(stop_words='english',binary=True,min_df=10)

XTrainBis=parseurBis.fit_transform(spamsTrain['message'])

#number of tokens

print('number of tokens : ',len(parseurBis.get_feature_names()))

#document term matrix

mdtTrainBis = XTrainBis.toarray()

#instatiate the object

modelBis= LogisticRegression()

#perform the training process

modelBis.fit(mdtTrainBis,spamsTrain['classe'])

#create the document term matrix for the test set

mdtTestBis=parseurBis.transform(spamsTest['message'])

#predection fot the test set

predTestBis=modelBis.predict(mdtTestBis)
#recall 

print(metrics.recall_score(spamsTest['classe'],predTestBis,pos_label='spam')) 
#precision 

print(metrics.precision_score(spamsTest['classe'],predTestBis,pos_label='spam'))
#F1-Score 

print(metrics.f1_score(spamsTest['classe'],predTestBis,pos_label='spam'))
#accuracy rate 

print(metrics.accuracy_score(spamsTest['classe'],predTestBis)) 