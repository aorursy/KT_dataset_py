# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import zipfile



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
archive_train = zipfile.ZipFile("/kaggle/input/whats-cooking/train.json.zip",'r')

archive_train
train_data = pd.read_json(archive_train.read('train.json'))

train_data
archive_test = zipfile.ZipFile("/kaggle/input/whats-cooking/test.json.zip",'r')

test_data = pd.read_json(archive_test.read("test.json"))

test_data
train_data.info()
test_data.info()
train_data['cuisine'].value_counts().shape
train_data["ingredients"][0]
#Storing distinct ingredients in a dict



train_ingr_count = {}

m = train_data.shape[0]

for i in range(m):

    for j in train_data["ingredients"][i]:

        if j in train_ingr_count.keys():

            train_ingr_count[j] += 1

        else:

            train_ingr_count[j] = 1

len(train_ingr_count)
#Storing distinct ingredients in a dict



test_ingr_count = {}

m = test_data.shape[0]

for i in range(m):

    for j in test_data["ingredients"][i]:

        if j in test_ingr_count.keys():

            test_ingr_count[j] += 1

        else:

            test_ingr_count[j] = 1

len(test_ingr_count)
# Adding ingredients from test dataset which aren't in train dataset and assigning them 0



train_ingr_missing = []



for i in test_ingr_count.keys():

    if i not in train_ingr_count.keys():

        train_ingr_missing.append(i)

        

print(len(train_ingr_missing))



for i in train_ingr_missing:

    train_ingr_count[i] = 0



print(len(train_ingr_count))
# Adding ingredients from train dataset which aren't in test dataset and assigning them 0



test_ingr_missing = []



for i in train_ingr_count.keys():

    if i not in test_ingr_count.keys():

        test_ingr_missing.append(i)

        

print(len(test_ingr_missing))



for i in test_ingr_missing:

    test_ingr_count[i] = 0



print(len(test_ingr_count))
for i in train_ingr_count.keys():

    train_data[i] = np.zeros(len(train_data))
train_data.head()
for i in test_ingr_count.keys():

    test_data[i] = np.zeros(len(test_data))
test_data.head()
print(train_data.shape)

print(test_data.shape)
for i in range(len(train_data)):

    for j in train_data['ingredients'][i]:

        train_data[j].iloc[i] = 1
train_data.head()
for i in range(len(test_data)):

    for j in test_data['ingredients'][i]:

        test_data[j].iloc[i] = 1
test_data.head()
train_data.drop('ingredients',axis=1,inplace=True)

test_data.drop('ingredients',axis=1,inplace=True)
train_data.columns
test_data = test_data[train_data.drop('cuisine',axis=1).columns]
test_data.columns
X = train_data.drop(['id','cuisine'],axis=1)

y = train_data['cuisine']



from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y)



print(X_test.shape,y_test.shape)

print(X_train.shape,y_train.shape)
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()

lr.fit(X_train,y_train)

lr.score(X_test,y_test)
pred = lr.predict(test_data.drop('id',axis=1))
submission = pd.DataFrame(data=pred,columns=['cuisine'])

submission['id'] = test_data['id']

submission.set_index("id",inplace=True)
submission.head()
submission.to_csv('submission.csv')