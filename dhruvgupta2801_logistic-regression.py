# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import zipfile



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
archive_train=zipfile.ZipFile('/kaggle/input/whats-cooking/train.json.zip','r')

train_data=pd.read_json(archive_train.read('train.json'))

train_data.head()
archive_test=zipfile.ZipFile('/kaggle/input/whats-cooking/test.json.zip','r')

test_data=pd.read_json(archive_test.read('test.json'))

test_data.head()
train_data.info()
test_data.info()
train_data['cuisine'].value_counts()
#train_data['clean_ingredients']=[' , '.join(z).strip() for z in train_data['ingredients']]

#test_data['clean_ingredients']=[' , '.join(z).strip() for z in test_data['ingredients']]
train_ingredients_count={}

for i in range(len(train_data)):

    for j in train_data['ingredients'][i]:

        if j in train_ingredients_count.keys():

            train_ingredients_count[j]+=1

        else:

            train_ingredients_count[j]=1
test_ingredients_count={}

for i in range(len(test_data)):

    for j in test_data['ingredients'][i]:

        if j in test_ingredients_count.keys():

            test_ingredients_count[j]+=1

        else:

            test_ingredients_count[j]=1
print(len(train_ingredients_count))

print(len(test_ingredients_count))
ingredients_missing_train=[]

for i in test_ingredients_count.keys():

    if i not in train_ingredients_count.keys():

        ingredients_missing_train.append(i)

print(len(ingredients_missing_train))        
for i in ingredients_missing_train:

    train_ingredients_count[i]=0

print(len(train_ingredients_count))    
ingredients_missing=[]

for i in train_ingredients_count.keys():

    if i not in test_ingredients_count.keys():

        ingredients_missing.append(i)

print(len(ingredients_missing))        
for i in ingredients_missing:

    test_ingredients_count[i]=0

print(len(test_ingredients_count))    
for i in train_ingredients_count.keys():

    train_data[i]=np.zeros(len(train_data))
for i in test_ingredients_count.keys():

    test_data[i]=np.zeros(len(test_data))
print(train_data.shape)

print(test_data.shape)
for i in range(len(train_data)):

    for j in train_data['ingredients'][i]:

        train_data[j].iloc[i]=1
train_data.head()
for i in range(len(test_data)):

    for j in test_data['ingredients'][i]:

        test_data[j].iloc[i]=1
test_data.head()
#test_data=test_data[train_data.drop(['cuisine'],axis=1).columns]
test_data=test_data[train_data.drop('cuisine',axis=1).columns]
from sklearn.model_selection import train_test_split

X=train_data.drop(['id','ingredients','cuisine'],axis=1)

y=train_data['cuisine']
X_train,X_val,y_train,y_val=train_test_split(X,y,random_state=42)
print(X_train.shape,y_train.shape)

print(X_val.shape,y_val.shape)
from sklearn.linear_model import LogisticRegression

from sklearn import preprocessing
lr=LogisticRegression()

lr.fit(X_train,y_train)
lr.score(X_val,y_val)
test_data['cuisine']=lr.predict(test_data.drop(['id','ingredients'],axis=1))
Submission=test_data[['id','cuisine']]

Submission.set_index('id',inplace=True)
Submission.to_csv('Submission.csv')