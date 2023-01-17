# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import zipfile as zf



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train_zip=zf.ZipFile('/kaggle/input/whats-cooking/train.json.zip','r')

train_data=pd.read_json(train_zip.read('train.json'))
test_zip=zf.ZipFile('/kaggle/input/whats-cooking/test.json.zip','r')

test_data=pd.read_json(test_zip.read('test.json'))
ing_train={}

for i in range(len(train_data)):

    for j in train_data['ingredients'][i]:

        if j in ing_train.keys():

            ing_train[j]+=1

        else:

            ing_train[j]=1

ing_test={}

for i in range(len(test_data)):

    for j in test_data['ingredients'][i]:

        if j in ing_test.keys():

            ing_test[j]+=1

        else:

            ing_test[j]=1
for i in ing_test.keys():

    if i not in ing_train.keys():

        ing_train[i]=0

for i in ing_train.keys():

    if i not in ing_test.keys():

        ing_test[i]=0
for i in ing_train.keys():

    train_data[i]=np.zeros(len(train_data))

for i in ing_test.keys():

    test_data[i]=np.zeros(len(test_data))
for i in range(len(train_data)):

    for j in train_data['ingredients'][i]:

        train_data[j].iloc[i]=1

for i in range(len(test_data)):

    for j in test_data['ingredients'][i]:

        test_data[j].iloc[i]=1
test_data=test_data[train_data.drop('cuisine',axis=1).columns]
from sklearn.model_selection import train_test_split

x=train_data.drop(['id','cuisine','ingredients'],axis=1)

y=train_data['cuisine']

X_train,X_test,y_train,y_test=train_test_split(x,y)
from sklearn.linear_model import LogisticRegression

lr=LogisticRegression()

lr.fit(X_train,y_train)

lr.score(X_test,y_test)
test_data['cuisine']=lr.predict(test_data.drop(['id','ingredients'],axis=1))
submission=test_data[['id','cuisine']]

submission.set_index('id',inplace=True)

submission.to_csv('submission.csv')