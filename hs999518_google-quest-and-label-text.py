import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

gg=pd.read_csv("../input/google-quest-challenge/train.csv")

ts=pd.read_csv("../input/google-quest-challenge/test.csv")

ss=pd.read_csv("../input/google-quest-challenge/sample_submission.csv")

ss

ss.to_csv("submission.csv",index=False)

ac=gg["answer_satisfaction"].value_counts()

ac

fig,ax=plt.subplots(figsize=(20,20))

ax.set_xticklabels(ac.index, size=15,rotation=90)

plt.scatter(ac.index,ac.values)

plt.plot(ac.index,ac.values)

plt.title("AnswerSatisfaction",size=50)

plt.show()

ac1=gg["question_body_critical"].value_counts()

ac1

from matplotlib import style

fig,ax=plt.subplots(figsize=(20,20))

ax.set_xticklabels(ac1.index, size=15,rotation=90)





style.use('ggplot') 



plt.plot(ac1.index,ac1.values,linewidth=10)

plt.title("question_body_critical",size=50)

plt.show()
ct=ts["category"].value_counts()

ct

fig,ax=plt.subplots(figsize=(10,10))

plt.bar(ct.index,ct.values)

plt.show()
ct1=ts["host"].value_counts()

ct1

fig,ax=plt.subplots(figsize=(20,20))

ax.set_xticklabels(ct1.index, size=15,rotation=90)

plt.scatter(ct1.index,ct1.values)

plt.plot(ct1.index,ct1.values)

plt.show()
xtrain=gg.iloc[:,[1,2]]

ytrain=gg.iloc[:,11]

xtest=ts.iloc[:,[1,2]]

from sklearn.preprocessing import LabelEncoder

le=LabelEncoder()

ytrain=le.fit_transform(ytrain)

ytrain
from sklearn.feature_extraction.text import CountVectorizer

vocabulary_count=CountVectorizer()

#print(len(vocabulary_count.vocabulary_))   

a= vocabulary_count.fit_transform(xtrain.iloc[:,0])





at=vocabulary_count.transform(xtest.iloc[:,0])



b= vocabulary_count.fit_transform(xtrain.iloc[:,1])



bt=vocabulary_count.transform(xtest.iloc[:,1])

a=a.toarray()

at=at.toarray()

b=b.toarray()

bt=bt.toarray()

a.shape

b.shape

at.shape

bt.shape

Xtr=np.hstack([a,b])

Xte=np.hstack([at,bt])

xtrain.shape


from sklearn.naive_bayes import MultinomialNB

model = MultinomialNB()

model.fit(Xtr,ytrain)





#print('predicted:',model.predict(titletest))

prediction=model.predict(Xte)

prediction

plt.hist(prediction)

plt.show()