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
train = pd.read_csv("/kaggle/input/titanic/train.csv")

test = pd.read_csv("/kaggle/input/titanic/test.csv")

test = test.fillna(0)



full_data = [train,test]



train.head(2)

test.head(2)
for data in full_data:

    data['Age'] = data['Age'].fillna(data['Age'].mean())

    data['family']=data['SibSp'] + data['Parch']

    data.drop(['SibSp','Parch'],axis=1,inplace=True)

    data.drop('Cabin',axis=1,inplace=True)

    data.fillna(data['Embarked'].value_counts().index[0])

    print(train.info())

    

    data['Sex']=data['Sex'].astype('category').cat.codes

    print(train.info())

   

    data['Embarked']=data['Embarked'].astype('category').cat.codes

    print(train.info())

    data.drop('Ticket',axis=1,inplace=True)

    data.drop('Name',axis=1,inplace=True)

    

for dataset in full_data:

    dataset['Age'] = dataset['Age'].fillna(data['Age'].mean())

    dataset.loc[dataset['Age'] <= 20, 'Age' ] = 0

    dataset.loc[(dataset['Age'] > 20) & (dataset['Age'] <= 30), 'Age' ] = 1

    dataset.loc[(dataset['Age'] > 30) & (dataset['Age'] <= 40), 'Age' ] = 2

    dataset.loc[(dataset['Age'] > 40) & (dataset['Age'] <= 50), 'Age' ] = 3

    dataset.loc[(dataset['Age'] > 50) & (dataset['Age'] <= 60), 'Age' ] = 4

    dataset.loc[(dataset['Age'] > 60) , 'Age' ] = 5

    dataset.loc[dataset['Age'].isnull(),'Age'] = 6;

    

for dataset in full_data:

    dataset.loc[dataset['Fare'] <= 7.91, 'Fare' ] = 0

    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare' ] = 1

    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare' ] = 2

    dataset.loc[(dataset['Fare'] > 31) , 'Fare' ] = 3

    dataset['Fare'] = dataset['Fare'].astype(int)

full_data[0:5]

train.head()
X_train = train.drop(['Survived'],axis=1)

Y_train = train['Survived']

X_test = test

print(X_train.shape,Y_train.shape)

print(X_test.shape)

print(X_train.columns)

print(X_test.columns)

X_train.fillna(0)

from sklearn.linear_model import LogisticRegression

log = LogisticRegression()



X_train[pd.isnull(X_train.Pclass)]

X_train[pd.isnull(X_train.Sex)]

X_train[pd.isnull(X_train.Age)]

X_train[pd.isnull(X_train.Fare)]

#X_train[pd.isnull(X_train.Embarked)]

#X_train[pd.isnull(X_train.family)]




log.fit(X_train,Y_train)

Y_pred = log.predict(X_test)

print(round(log.score(X_train,Y_train)*100))

# print(Y_pred)
PassengerId = np.array(X_test["PassengerId"]).astype(int)

# my_prediction(予測データ）とPassengerIdをデータフレームへ落とし込む

my_solution = pd.DataFrame(Y_pred, PassengerId, columns = ["Survived"])

# my_tree_one.csvとして書き出し

my_solution.to_csv("my_first_charenge.csv", index_label = ["PassengerId"])