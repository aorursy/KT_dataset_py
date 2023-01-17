# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train = pd.read_csv('/kaggle/input/titanic/train.csv')

test = pd.read_csv('/kaggle/input/titanic/test.csv')



train.head()
train.info()

import matplotlib.pyplot as plt

plt.figure(figsize=(15,10))

plt.hist(train['Age'])
plt.figure(figsize=(15,10))

train[train['Survived'] == 1]['Fare'].hist()

train.nlargest(5, 'Age').filter(items=['Age', 'Name', 'Pclass'])
from sklearn.neighbors import KNeighborsClassifier
train
test.isna().sum()
train.Age = train.Age.fillna(train.Age.mean())

test.Age = test.Age.fillna(test.Age.mean())

test.Fare  = test.Fare.fillna(test.Fare.mean())
model = KNeighborsClassifier(n_neighbors=20)

features = ['Parch','Age','Fare']



model.fit(train[features],train['Survived'])



predictions = model.predict(test[features])
Pid = test['PassengerId']



#set the output as a dataframe and convert to csv file named submission.csv

output = pd.DataFrame({ 'PassengerId' : Pid, 'Survived': predictions })

output.to_csv('submission.csv', index=False)