import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns



import warnings

warnings.filterwarnings('ignore')



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# Reading data

train_data = pd.read_csv('/kaggle/input/titanic/train.csv')

test_data = pd.read_csv('/kaggle/input/titanic/test.csv')



# Storing Passenger Id for submission

Id = test_data.PassengerId
dataset = pd.concat([train_data, test_data], sort=False, ignore_index=True)
# 使用平均年龄来填充年龄中的nan值

dataset['Age'].fillna(train_data['Age'].mean(), inplace=True)



# 使用票价的均值填充票价中的nan值

dataset['Fare'].fillna(train_data['Fare'].mean(), inplace=True)



# 使用登录最多的港口来填充登录港口的nan值

dataset['Embarked'].fillna('S', inplace=True)

# 增加一个title字段

dataset['Title'] = dataset['Name'].str.extract(' ([A-Za-z]+)\.', expand = False)



dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')

dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')

dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

    

# map 替换

title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}

dataset['Title'] = dataset['Title'].map(title_mapping)



# Imputing missing values with 0

dataset['Title'] = dataset['Title'].fillna(0)

# dataset.head()
# Family Size = # of Siblings + # of Parents + You

dataset['FamSize'] = dataset['SibSp'] + dataset['Parch'] + 1
# 特征选择

# 删除useless

dataset.drop(['Ticket'], axis=1, inplace=True)

dataset.drop(['Name'], axis=1, inplace=True)

dataset.drop(['Cabin'], axis=1, inplace=True)

dataset.drop(['PassengerId'], axis=1, inplace=True)

dataset.drop(['FamSize'], axis=1, inplace=True)



# Splitting dataset into train

train_data = dataset[:len(train_data)]



# Splitting dataset into test

test_data = dataset[len(train_data):]



# Drop labels 'Survived' because there shouldn't be a Survived column in the test data

test_data.drop(labels=['Survived'], axis=1, inplace=True)
train_data['Survived'] = train_data['Survived'].astype(int)

y=train_data.Survived

X=train_data.drop('Survived', axis=1)

test_features = test_data
from sklearn.feature_extraction import DictVectorizer

dvec=DictVectorizer(sparse=False)

X = dvec.fit_transform(X.to_dict(orient='record'))
# 查看特征矩阵

print(dvec.feature_names_)
# 模型



from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_score

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.naive_bayes import GaussianNB 

from sklearn.ensemble import RandomForestClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.ensemble import GradientBoostingRegressor
cross_val_score(DecisionTreeClassifier(criterion='gini', ccp_alpha=0.004), X, y).mean()
cross_val_score(RandomForestClassifier(), X, y).mean()
cross_val_score(GradientBoostingClassifier(), X, y).mean()
final_model = DecisionTreeClassifier(criterion='gini', ccp_alpha=0.004)



# Train final_model with train data

final_model.fit(X, y)



# Treat the test data in the same way as training data. In this case, pull same columns.

test_features=dvec.transform(test_features.to_dict(orient='record'))

# 决策树预测

pred_labels = final_model.predict(test_features)



print(pred_labels)
output = pd.DataFrame({'PassengerId': Id, 'Survived':pred_labels})

output.to_csv('gender_submission.csv', index=False)
output.head()