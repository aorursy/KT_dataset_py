import numpy as np

import pandas as pd 

from sklearn.ensemble import RandomForestClassifier



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
train_df = pd.read_csv("/kaggle/input/titanic/train.csv")

#train_df.head()



test_df = pd.read_csv("/kaggle/input/titanic/test.csv")

#test_df.head()



traintest_df = [train_df, test_df]
print(train_df.columns.values)
women = train_df.loc[train_df.Sex == 'female']["Survived"]

rate_women = sum(women)/len(women)



print("Percent of women who survived:", rate_women)
men = train_df.loc[train_df.Sex == 'male']["Survived"]

rate_men = sum(men)/len(men)



print("Percent of men who survived:", rate_men)
train_df.describe(include=['O'])
train_df[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)
for dataset in traintest_df:

    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)



pd.crosstab(train_df['Title'], train_df['Sex'])
for dataset in traintest_df:

    dataset['Title'] = dataset['Title'].replace(['Capt', 'Col','Countess','Lady','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Other')



    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

    

train_df[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()
title_clear = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Other": 5}

for dataset in traintest_df:

    dataset['Title'] = dataset['Title'].map(title_clear)

    dataset['Title'] = dataset['Title'].fillna(0)



train_df.head()
train_df = train_df.drop(['Name', 'PassengerId'], axis=1)

test_df = test_df.drop(['Name'], axis=1)

traintest_df = [train_df, test_df]

train_df.shape, test_df.shape
for dataset in traintest_df:

    dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0} ).astype(int)



train_df.head()
train_df = train_df.drop(['Age', 'SibSp', 'Parch','Ticket', 'Fare','Cabin','Embarked'], axis=1)

traintest_df = [train_df, test_df]

train_df.head()
test_df = test_df.drop(['Age', 'SibSp', 'Parch','Ticket', 'Fare','Cabin','Embarked'], axis=1)

traintest_df = [train_df, test_df]

test_df.head()
df1_train = train_df.drop("Survived", axis=1)

df2_train = train_df["Survived"]

df1_test  = test_df.drop("PassengerId", axis=1).copy()

df1_train.shape, df2_train.shape, df1_test.shape
random_forest_1 = RandomForestClassifier(n_estimators=100)

random_forest_1.fit(df1_train, df2_train)

df2_pred = random_forest_1.predict(df1_test)

random_forest_1.score(df1_train, df2_train)

rate_random_forest = round(random_forest_1.score(df1_train, df2_train) * 100, 2)

rate_random_forest
my_submission = pd.DataFrame({"PassengerId": test_df["PassengerId"],"Survived": df2_pred})

my_submission.to_csv('my_submission_AG5.csv', index=False)

print('Submission was successfully!')