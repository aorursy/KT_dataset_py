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
train_data = pd.read_csv("/kaggle/input/titanic/train.csv")

train_data.head()
test_data = pd.read_csv("/kaggle/input/titanic/test.csv")

test_data.head()
test_data.drop('Cabin',inplace=True,axis=1)

train_data.drop('Cabin',inplace=True,axis=1)
import seaborn as sns

sns.heatmap(train_data.isnull())
def impute_data(data):

    if pd.isnull(data['Age']):

        if data['Pclass'] == 1:

            return 37



        elif data['Pclass'] == 2:

            return 29



        else:

            return 24



    else:

        return data['Age']
train_data['Age']=train_data[['Age','Pclass']].apply(impute_data,axis=1)
test_data['Age']=test_data[['Age','Pclass']].apply(impute_data,axis=1)
cols=['Sex','Embarked']

for i in cols:

    i=pd.get_dummies(train_data[i],drop_first=True)

    train_data=pd.concat([i,train_data],axis=1)

train_data.drop('Sex',axis=1,inplace=True)
cols=['Sex','Embarked']

for i in cols:

    i=pd.get_dummies(test_data[i],drop_first=True)

    test_data=pd.concat([i,test_data],axis=1)

test_data.drop('Sex',axis=1,inplace=True)

test_data.drop('Embarked',axis=1,inplace=True)

train_data.drop('Embarked',axis=1,inplace=True)

train_data.head()

test_data.head()
features=['Q','S','male','Age','Parch','SibSp']

X_train=train_data[features]

y_train=train_data['Survived']

X_test=test_data[features]
from sklearn.linear_model import LogisticRegression
log=LogisticRegression()
log.fit(X_train,y_train)
predictions=log.predict(X_test)
output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})

output.to_csv('my_submission.csv', index=False)

output.head()