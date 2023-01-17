# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from sklearn import preprocessing

from sklearn.linear_model import LogisticRegression

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train_data = pd.read_csv("/kaggle/input/titanic/train.csv",index_col = 0)



train_data.head()

test_data = pd.read_csv("/kaggle/input/titanic/test.csv")

test_data.head()
test_data.info()

test_cabin = test_data['Cabin']

test_cabin.fillna(0,inplace=True)

test_data.head()
test_data['Cabin'].describe()
train_data.isnull()
cabin = train_data['Cabin']

cabin.fillna(0,inplace=True)

train_data.head()
scaler = preprocessing.MinMaxScaler()

train_data[['Pclass','Age','Fare']] = scaler.fit_transform(train_data[['Pclass','Age','Fare']])

train_data.head()

test_data[['Pclass','Age','Fare']] = scaler.fit_transform(test_data[['Pclass','Age','Fare']])

test_data.head()
def replace(x):

    if x != 0 :

        x=1

    return x



train_data['Cabin']=train_data['Cabin'].apply(replace)

test_data['Cabin'] = test_data['Cabin'].apply(replace)

test_data.head()
cabin = train_data['Cabin']

cabin.fillna(0,inplace=True)

train_data.head()
def sex(x):

    if x == 'male':

        return 0

    else :

        return 1



train_data['Sex'] = train_data['Sex'].apply(sex)

test_data['Sex'] = test_data['Sex'].apply(sex)

train_data.head()
features = ['Pclass','Sex','Age','Parch','SibSp','Fare','Cabin']

y = train_data['Survived']

X_train = train_data[features]

X_test = test_data[features]
X_train.head()
X_test.head()
X_test['Cabin'].fillna(0,inplace=True);

X_test.head()
X_test['Cabin'] = X_test['Cabin'].apply(replace)

X_test['Cabin'].describe()

X_test.head()


X_train.fillna(0,inplace=True)

X_train.info()

X_test.fillna(0,inplace=True)

X_test.info()
model = LogisticRegression()

model.fit(X_train,y)

predictions = model.predict(X_test)

output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})

output.to_csv('my_submission.csv', index=False)

print("Your submission was successfully saved!")