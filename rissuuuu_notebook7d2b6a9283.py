# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import re
import nltk
from nltk.corpus import stopwords
import seaborn as sns
import matplotlib.pyplot as pl
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import os
import glob
files='../input/bbc-full-text-document-classification/bbc/'
os.listdir(files)
from collections import defaultdict
dicts=defaultdict(list)

for dir_name,_,file_names in os.walk(files):
    for file in file_names:
        dicts['categories'].append(os.path.basename(dir_name))
        name=os.path.splitext(file)[0]
        dicts['doc_id'].append(name)
        path=os.path.join(dir_name,file)
        
        with open(path,'r',encoding='latin-1') as file:
            dicts['text'].append(file.read())
        

df=pd.DataFrame.from_dict(dicts)
df.head(10)
df.drop(0,inplace=True)
df.reset_index(drop=True,inplace=True)
df.drop('doc_id',axis=1,inplace=True)
import random
print(df['text'][1])
random.sample(range(df.text.shape[0]),5)
ax=sns.countplot(df['categories'])
for i,j in enumerate(df['categories'].value_counts().values):
    ax.text(i-0.3,100,j,fontsize=20)
pl.title('Value counts for each categories')
pl.show()
' '.join( df['text'][0].split('\n')[1:])
' '.join( df['text'][0].split('\n'))
def returnTitle(data):
    data=data.split('\n')[0]
    return data
def returnArticle(data):
    data=' '.join(data.split('\n')[1:])
    return data

    
df['title']=df['text'].apply(lambda x:returnTitle(x))
df['article']=df['text'].apply(lambda x: returnArticle(x))
print(returnArticle('My name is\n Rishav Paudel\n I read in Hyderabad'))
df.head()
wordlength=dict(df['title'].str.split().apply(len).value_counts())
pl.bar(list(wordlength.keys()),list(wordlength.values()))
pl.xlabel('Length of word document')
pl.ylabel('Documents Count')
pl.title('Number of words in each document')
pl.show()
import nltk

words={}
for i in df['title']:
    for j in i.split():
        words[j]=0
for i in df['title']:
    for j in i.split():
        words[j]+=1
toptitles=nltk.FreqDist(words)
words=list(toptitles.keys())[:15]
vals=list(toptitles.values())[:15]
pl.figure(figsize=(8,10))
ax=sns.barplot(vals,words)
for i,j in enumerate(vals):
    ax.text(8,i+.2,j,fontsize=15)
    
pl.title('Maximum occurances of word in title')
pl.show()
categories=defaultdict(list)
for i in df['categories'].unique():
    temp=df[df['categories']==i]['article'].str.split().apply(len).values
    categories[i]=temp
pl.boxplot(categories.values())

pl.xticks(range(5),categories.keys())
pl.show()

for i in categories.keys():
    pl.plot(categories[i])
    pl.title(i)
    pl.show()
def Clean(text):
    text=text.split()
    text=[i.lower() for i in text if i.lower() not in stopwords.words('english')]
    text=' '.join(text)
    text=re.sub('[^A-Za-z0-9]+',' ',text)
    text=text.lower()
    return text
df['article']=df['article'].apply(lambda x: Clean(x))
df.head()
from sklearn.preprocessing import LabelEncoder
l_enc=LabelEncoder()
df['categories']=l_enc.fit_transform(df['categories'])
df.head()
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf=TfidfVectorizer()
X=tfidf.fit_transform(df['article'])
y=df['categories']
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.3)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.linear_model import PassiveAggressiveClassifier
models=[LogisticRegression(),RandomForestClassifier(),DecisionTreeClassifier(),
       MultinomialNB(),XGBClassifier(),SVC(),PassiveAggressiveClassifier()]
scores=pd.DataFrame({'Model':[],'Train_Score':[],'Test_Score':[]})
for j,i in enumerate(models):
    i.fit(X_train,y_train)
    scores.loc[j,:]=[i,i.score(X_train,y_train),i.score(X_test,y_test)]
scores
















































































