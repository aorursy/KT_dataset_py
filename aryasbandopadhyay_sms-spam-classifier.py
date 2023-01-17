import os 
from collections import Counter
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
data = pd.read_csv("../input/spam.csv",encoding='latin-1')
data.head()
data['Unnamed: 2'].count()
data['Unnamed: 3'].count()
data['Unnamed: 4'].count()
data.drop(columns=['Unnamed: 2','Unnamed: 3','Unnamed: 4'],inplace=True)
data.head()
data.rename(columns={'v1':'Label','v2':'Message'},inplace=True)
data
data.dropna(inplace=True)
data.shape
words=[]

for i in range(len(data['Message'])):
    blob=data['Message'][i]
    words+=blob.split(" ")
len(words)
for i in range(len(words)):
    if not words[i].isalpha():
        words[i]=""
word_dict=Counter(words)
word_dict
len(word_dict)
del word_dict[""]
word_dict=word_dict.most_common(3000)
features=[]
labels=[]

for i in range(len(data['Label'])):

    blob=data['Message'][i].split(" ")
    data1=[]
    for j in word_dict:
        data1.append(blob.count(j[0]))
    features.append(data1)
    
    
   
    
    
features=np.array(features)
features.shape
labels=data.iloc[:,0]
for i in range(len(labels)):
    if labels[i]=='ham':
        labels[i]=0
    else:
        labels[i]=1
labels.shape
labels=labels.values
labels=labels.astype(int)
labels
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(features,labels,test_size=0.2,random_state=9)
xtrain.shape
from sklearn.naive_bayes import MultinomialNB
nbs=MultinomialNB()
nbs.fit(xtrain,ytrain)
y_pred=nbs.predict(xtest)
from sklearn.metrics import accuracy_score
accuracy_score(y_pred,ytest)
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(xtrain,ytrain)
pred = model.predict(xtest)
accuracy_score(ytest,pred)
from sklearn.ensemble import RandomForestClassifier
model1= RandomForestClassifier()
model1.fit(xtrain,ytrain)
prediction = model1.predict(xtest)
accuracy_score(ytest,prediction)
from sklearn.metrics import confusion_matrix,classification_report
print(classification_report(ytest, y_pred, target_names = ["Ham", "Spam"]))
print(classification_report(ytest, pred, target_names = ["Ham", "Spam"]))

conf_mat = confusion_matrix(ytest, y_pred)
conf_mat_normalized = conf_mat.astype('float') / conf_mat.sum(axis=1)[:, np.newaxis]
sns.heatmap(conf_mat_normalized)
plt.ylabel('True label')
plt.xlabel('Predicted label')
print(conf_mat)


