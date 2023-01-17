# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.naive_bayes import GaussianNB,MultinomialNB,BernoulliNB

from sklearn.metrics import  accuracy_score,classification_report

from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import TfidfVectorizer

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data= pd.read_csv("../input/spam.csv",encoding='latin-1')

data.head()
columns=["type","emails","col1","col2","col3"]

data.columns=columns

data.head()
data.drop(columns=['col1','col2','col3'],axis=1,inplace=True)

data.head()
X=data.emails

y=data.type

vectoriser=TfidfVectorizer()

X=vectoriser.fit_transform(X)

feature_Names=vectoriser.get_feature_names()

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=10)
model = BernoulliNB()

model.fit(X_train,y_train)

y_predict = model.predict(X_test)   
print(accuracy_score(y_test,y_predict))

print(classification_report(y_test,y_predict))

pd.crosstab(y_test,y_predict)