# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.linear_model import LogisticRegression
data = pd.read_csv('../input/Combined_News_DJIA.csv')

data.head(1)
train = data[data['Date'] < '20150101']

test = data[data['Date'] > '20141231']
# Removing punctuations

slicedData= train.iloc[:,2:27]

slicedData.replace(to_replace="[^a-zA-Z]", value=" ", regex=True, inplace=True)



# Renaming column names for ease of access

list1= [i for i in range(25)]

new_Index=[str(i) for i in list1]

slicedData.columns= new_Index

slicedData.head(5)



# Convertng headlines to lower case

for index in new_Index:

    slicedData[index]=slicedData[index].str.lower()

slicedData.head(1)
headlines = []

for row in range(0,len(slicedData.index)):

    headlines.append(' '.join(str(x) for x in slicedData.iloc[row,0:25]))
headlines[0]
basicvectorizer = CountVectorizer(ngram_range=(1,1))

basictrain = basicvectorizer.fit_transform(headlines)

print(basictrain.shape)
# model 1: logistic regression

# 1-gram

basicmodel = LogisticRegression()

basicmodel = basicmodel.fit(basictrain, train["Label"])
testheadlines = []

for row in range(0,len(test.index)):

    testheadlines.append(' '.join(str(x) for x in test.iloc[row,2:27]))

basictest = basicvectorizer.transform(testheadlines)

predictions = basicmodel.predict(basictest)
predictions
pd.crosstab(test["Label"], predictions, rownames=["Actual"], colnames=["Predicted"])
from sklearn.metrics import classification_report

from sklearn.metrics import f1_score

from sklearn.metrics import accuracy_score 

from sklearn.metrics import confusion_matrix



print (classification_report(test["Label"], predictions))

print (accuracy_score(test["Label"], predictions))
# 2-gram

basicvectorizer2 = CountVectorizer(ngram_range=(1,2))

basictrain2 = basicvectorizer2.fit_transform(headlines)

print(basictrain2.shape)



basicmodel2 = LogisticRegression()

basicmodel2 = basicmodel2.fit(basictrain2, train["Label"])



basictest2 = basicvectorizer2.transform(testheadlines)

predictions2 = basicmodel2.predict(basictest2)



pd.crosstab(test["Label"], predictions2, rownames=["Actual"], colnames=["Predicted"])



print (classification_report(test["Label"], predictions2))

print (accuracy_score(test["Label"], predictions2))

print (classification_report(test["Label"], predictions2))

print (accuracy_score(test["Label"], predictions2))
# 3-gram

basicvectorizer3 = CountVectorizer(ngram_range=(2,3))

basictrain3 = basicvectorizer3.fit_transform(headlines)

print(basictrain3.shape)



basicmodel3 = LogisticRegression()

basicmodel3 = basicmodel3.fit(basictrain3, train["Label"])



basictest3 = basicvectorizer3.transform(testheadlines)

predictions3 = basicmodel3.predict(basictest3)



pd.crosstab(test["Label"], predictions3, rownames=["Actual"], colnames=["Predicted"])



print (classification_report(test["Label"], predictions3))

print (accuracy_score(test["Label"], predictions3))