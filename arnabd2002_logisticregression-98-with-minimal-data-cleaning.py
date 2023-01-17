# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import re

import seaborn as sns

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
realFile='/kaggle/input/fake-and-real-news-dataset/True.csv'

fakeFile='/kaggle/input/fake-and-real-news-dataset/Fake.csv'
realDF=pd.read_csv(realFile)

fakeDF=pd.read_csv(fakeFile)
def formatDate(dateStr):

    try:

        return pd.to_datetime(dateStr)

    except Exception as e:

        return 'NA'
fakeDF['date']=fakeDF['date'].apply(lambda x:formatDate(x))

realDF['date']=realDF['date'].apply(lambda x:formatDate(x))
fakeDF.drop(index=fakeDF[fakeDF['date']=='NA'].index,inplace=True)
fakeDF['LABEL']=0

realDF['LABEL']=1
combinedDF=pd.concat([realDF,fakeDF])
combinedDF.columns
def getCleanText(txt):

    #print(txt)

    if txt is not None or txt!='':

        words=[re.sub(pattern='[^a-zA-Z0-9]',string=x,repl=' ').lower() for x in txt.split(' ') if x.isalpha() and x!=' ']

        return ' '.join(words)

    else:

        return None
combinedDF['Clean_text']=combinedDF['text'].apply(lambda x:getCleanText(x))
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.metrics import classification_report,confusion_matrix
tVec=TfidfVectorizer()
X=tVec.fit_transform(combinedDF['Clean_text'])

y=combinedDF['LABEL']
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split
lr=LogisticRegression()
X_train,X_test,y_train,y_test=train_test_split(X,y,train_size=0.85,shuffle=True,random_state=42)
lr=LogisticRegression()

lr.fit(X_train,y_train)
predictions=lr.predict(X_test)
lr.score(X_test,y_test)
print(classification_report(y_true=y_test,y_pred=predictions))
print(confusion_matrix(y_test,predictions))
sns.heatmap(confusion_matrix(y_test,predictions))