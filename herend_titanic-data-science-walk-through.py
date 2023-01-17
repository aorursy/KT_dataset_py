#import necessary packages and modules

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns
#bring in data

train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
#explore initial dimensions

train.head(5)
train.info()
train = train.drop(train[['Cabin','Embarked', 'Name', 'Ticket']], axis=1)

test = test.drop(test[['Cabin','Embarked', 'Name', 'Ticket']], axis=1)

train.dropna(axis=0, how='any', inplace=True)
train.head()
corr = train.corr()

sns.heatmap(corr, xticklabels=corr.columns.values, yticklabels=corr.columns.values)
train.drop(train[['Fare','SibSp', 'Parch']], axis=1, inplace=True)

test.drop(test[['Fare', 'SibSp', 'Parch']], axis=1, inplace=True)

train.head()
sex_variable = train.pivot_table(index='Sex', values='Survived')

sex_variable.plot.bar()

plt.show()
survived = train[train["Survived"] == 1]

died = train[train["Survived"] == 0]

survived["Age"].plot.hist(alpha=0.5,color='red',bins=50)

died["Age"].plot.hist(alpha=0.5,color='blue',bins=50)

plt.legend(['Survived','Died'])

plt.show()
def category_age(df,cut_points,label_names):

    df["Age"] = df["Age"].fillna(-0.5)

    df["Age_categories"] = pd.cut(df["Age"],cut_points,labels=label_names)

    return df



cut_points = [0,5,12,18,35,60,100]

label_names = ["Infant","Child","Teenager","Young Adult","Adult","Senior"]



train = category_age(train,cut_points,label_names)

test = category_age(test,cut_points,label_names)



pivot = train.pivot_table(index="Age_categories",values='Survived')

pivot.plot.bar()

plt.show()
def create_dummies(df,column_name):

    dummies = pd.get_dummies(df[column_name],prefix=column_name)

    df = pd.concat([df,dummies],axis=1)

    return df



for column in ["Pclass","Sex","Age_categories"]:

    train = create_dummies(train,column)

    test = create_dummies(test,column)
train = train.drop(train[['Pclass', 'Sex', 'Age', 'Age_categories']], axis=1)

test = test.drop(test[['Pclass', 'Sex', 'Age', 'Age_categories']], axis=1)
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import cross_val_score
lr = LogisticRegression()



train_y = train['Survived']

train_x = train.drop('Survived', axis=1)



scores = cross_val_score(lr, train_x, train_y, cv=10)

scores.sort()

accuracy = scores.mean()

print(scores)

print(accuracy)
lr.fit(train_x, train_y)

predictions = lr.predict(test)
ids = test['PassengerId']

submission_df = {'PassengerId': ids, 'Survived': predictions}
submission = pd.DataFrame(submission_df)

submission.to_csv('...submission.csv', index=False)