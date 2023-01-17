import os
import pandas as pd
import numpy as np
from subprocess import check_output
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
train = pd.read_csv("../input/train.csv")
test=pd.read_csv("../input/test.csv")
train.head()
train.tail()
test.head()
test.head()
print('Train data shape: ',train.shape)
print('Test data shape: ',test.shape)
train.isnull().sum()
print(train.info())
print("Length of unique id's in train: ",len(np.unique(train['id'])))
print("Length of train dataframe is: ",len(train))
id = test['id'].copy()
train = train.drop('id', axis = 1)
train['author'] = train.author.map({'EAP':0,'HPL':1,'MWS':2})
train.head()
x = train['text'].copy()
y = train['author'].copy()

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3, random_state=42)

print(x.head())
print(y.head())
# Example
# In short it returns the count of each word in row under consideration.

text = ["My name is Bhagwat Chate Bhagwat Chate"]
toy  = CountVectorizer(lowercase=False, token_pattern = r'\w+|\,')
toy.fit_transform(text)

print (toy.vocabulary_)
matrix = toy.transform(text)

print(matrix[0,0])
print(matrix[0,1])
print(matrix[0,2])
print(matrix[0,3])
print(matrix[0,4])
vect = CountVectorizer(lowercase=False, token_pattern=r'\w+|\,')

x_v = vect.fit_transform(x)
x_train_v = vect.transform(x_train)
x_test_v = vect.transform(x_test)

print (x_train_v.shape)
model = MultinomialNB()
model.fit(x_train_v, y_train)
print('Naive Bayes accuracy: ',round(model.score(x_test_v, y_test)*100,2),'%')
x_test=vect.transform(test["text"])
model = MultinomialNB()
model.fit(x_v, y)

predicted_result = model.predict_proba(x_test)

predicted_result.shape
result=pd.DataFrame()

result["id"]  = test['id']
result["EAP"] = predicted_result[:,0]
result["HPL"] = predicted_result[:,1]
result["MWS"] = predicted_result[:,2]

result.head()
result.to_csv("Result.csv", index=False)