# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd 

import numpy as np 

import re 

import nltk 

from nltk.corpus import stopwords 

from nltk.stem.porter import PorterStemmer 

from sklearn.feature_extraction.text import CountVectorizer 

from sklearn.model_selection import train_test_split 

from sklearn.tree import DecisionTreeClassifier

from sklearn.linear_model import LogisticRegression
nltk.download("stopwords")

train=pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')

test=pd.read_csv('/kaggle/input/nlp-getting-started/test.csv')

train.head()
train.isnull().sum().sort_values(ascending=False)

test.isnull().sum().sort_values(ascending=False)
train=train.drop(["location","keyword","id"] , axis=1)

test= test.drop(["location","keyword","id"] , axis=1)

train.columns

train["text"].head()

test.shape
corpus=[]

length=len(train["text"])

for i in range(length):

    

    tweet=re.sub('[^a-zA-z]'," " ,train["text"][i])

    tweet=tweet.lower()

    tweet=tweet.split()

    ps=PorterStemmer()

    tweet=[ ps.stem(word) for word in tweet  if not word in set(stopwords.words('english'))]

    tweet=" ".join(tweet)



    corpus.append(tweet)
cv=CountVectorizer()    

X=cv.fit_transform(corpus).toarray()
X.shape
X_test=cv.transform(test["text"]).toarray()
X_test.shape
y=train["target"]





print("x shape is :", X.shape)

print("y shape is :", y.shape)



X_train,X_valid,y_train,y_valid=train_test_split(X,y,test_size=.2 , random_state=33 , shuffle=True)

print("x_train.shape is: ",X_train.shape)

print("y_train.shape is: ",y_train.shape)
lr = LogisticRegression()  



lr.fit(X_train,y_train)
y_valid=lr.predict(X_valid)
y_valid
decisionTreeModel = DecisionTreeClassifier(criterion= 'entropy',max_depth = 10,max_leaf_nodes=30, random_state=55)



decisionTreeModel.fit(X_train,y_train)
y_valid=decisionTreeModel.predict(X_valid)
models=[decisionTreeModel,lr]

for model in models:

    print(type(model ) ,"train score is :" ,model.score(X_train , y_train) )

    print(type(model ) ,"test score is :" ,model.score(X_valid , y_valid) )

    y_pred=model.predict(X_test)

    print(type(model ) ,"test score is :" ,model.score(X_test , y_pred) )

    

y_pred
output = pd.DataFrame({'target': y_pred})

output.to_csv('submission.csv', index=False)
output.head()