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

from sklearn import preprocessing

from sklearn.tree import DecisionTreeClassifier
data = pd.read_csv('../input/train.csv')
data.head()
age = 29.699118
data['Embarked'] = data['Embarked'].fillna('D')

data['Pclass'] = data['Pclass'].fillna(2)



data['Age'] = data['Age'].fillna(age)

#data['Age'] = data['Age'].astype('float64')

# preprocessing

X = data[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Embarked', 'Fare']].values

X[:5]

data


X.shape

y = data['Survived']

data.dtypes
'''

now preparing for optimized computation I have to process this



'''

passanger_sex = preprocessing.LabelEncoder()

passanger_sex.fit(['female','male'])

X[:, 1] = passanger_sex.transform(X[:, 1])



X[:5]

#data['Fare'].dropna()
passenger_emb = preprocessing.LabelEncoder()

passenger_emb.fit(['S', 'C', 'Q', 'D'])

X[:, 5] = passenger_emb.transform(X[:, 5])

from sklearn.model_selection import train_test_split

xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size = .3, random_state = 3)

print(xtrain.shape, ytrain.shape)

print(xtest.shape, ytest.shape)
titanic = DecisionTreeClassifier(criterion='entropy', max_depth = 4)

titanic

titanic.fit(xtrain, ytrain)
prediction = titanic.predict(xtest)
#print(prediction[:5])

#print(ytest)
from sklearn import metrics

print("Prediction Accuracy: ", metrics.accuracy_score(ytest, prediction))