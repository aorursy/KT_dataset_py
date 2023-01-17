# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

dfs = [train, test]
for df in dfs:

    df['Sex'] = df['Sex'].map({'female': 1, 'male': 0})
train.head()
train.dtypes
for df in dfs:

    df.dropna()
for df in dfs:

    df[df.Cabin.notnull()]
train.head()
t = train.loc[0, 'Cabin']
type(t)
train.dtypes
test = test.drop(['Ticket', 'Name', 'Cabin'], axis=1)

train = train.drop(['Ticket', 'Name', 'Cabin'], axis=1)
train.head()
for df in dfs:

    df = pd.get_dummies(df, columns=['Embarked'])
for df in dfs:

    age_avg = df['Age'].mean()

    age_std = df['Age'].std()

    age_null_count = df['Age'].isnull().sum()

    

    age_null_random_list = np.random.randint(age_avg - age_std, age_avg + age_std, size=age_null_count)

    df['Age'][np.isnan(df['Age'])] = age_null_random_list

    df['Age'] = df['Age'].astype(int)
train.head()
for df in dfs:

    df.loc[ df['Age'] <= 16, 'Age'] 					       = 0

    df.loc[(df['Age'] > 16) & (df['Age'] <= 32), 'Age'] = 1

    df.loc[(df['Age'] > 32) & (df['Age'] <= 48), 'Age'] = 2

    df.loc[(df['Age'] > 48) & (df['Age'] <= 64), 'Age'] = 3

    df.loc[ df['Age'] > 64, 'Age'] = 4
train.dtypes
for df in dfs:

    df['Fare'] = df['Fare'].fillna(train['Fare'].median())
for df in dfs:

    df.loc[ df['Fare'] <= 7.91, 'Fare'] = 0

    df.loc[(df['Fare'] > 7.91) & (df['Fare'] <= 14.454), 'Fare'] = 1

    df.loc[(df['Fare'] > 14.454) & (df['Fare'] <= 31), 'Fare'] = 2

    df.loc[ df['Fare'] > 31, 'Fare'] = 3

    df['Fare'] = df['Fare'].astype(int)
train.head()
train.head()
test.head()
test = pd.get_dummies(test, columns=['Embarked'])

train = pd.get_dummies(train, columns=['Embarked'])
test.head()
test = test.drop(['PassengerId'], axis=1)
test = test.dropna()

train = train.dropna()
test.head()

train = train.drop(['PassengerId'], axis=1)
train.head()
train.dtypes
test.dtypes
test_data = test.values

train_data = train.values



from sklearn.ensemble import RandomForestClassifier

forest = RandomForestClassifier(n_estimators = 100)

forest = forest.fit(train_data[0::,1::],train_data[0::,0])

output = forest.predict(test_data)
print(output)
test_results = test

test_results['SurvivalGuess'] = output
test_results.head()
len(output)