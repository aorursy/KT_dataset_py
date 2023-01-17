# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.ensemble import RandomForestClassifier
# get titanic & test csv files as a DataFrame

train_df = pd.read_csv("../input/train.csv")



# Drop useless columns

train_df = train_df.drop("Name", 1)

train_df = train_df.drop("Ticket", 1)

train_df = train_df.drop("Cabin", 1)



# Change strings by int

train_df['Sex'] = train_df['Sex'].map( {'female': 0, 'male': 1} ).astype(int)

if len(train_df.Embarked[ train_df.Embarked.isnull() ]) > 0:

    train_df.Embarked[ train_df.Embarked.isnull() ] = train_df.Embarked.dropna().mode().values

train_df['Embarked'] = train_df['Embarked'].map( {'C': 0, 'Q': 1, 'S':2} ).astype(int)



# preview the data

train_df.head()
# get titanic & test csv files as a DataFrame

test_df    = pd.read_csv("../input/test.csv")



# Drop useless columns

test_df = test_df.drop("Name", 1)

test_df = test_df.drop("Ticket", 1)

test_df = test_df.drop("Cabin", 1)



# Change strings by int

test_df['Sex'] = test_df['Sex'].map( {'female': 0, 'male': 1} ).astype(int)

if len(test_df.Embarked[ test_df.Embarked.isnull() ]) > 0:

    test_df.Embarked[ test_df.Embarked.isnull() ] = test_df.Embarked.dropna().mode().values

test_df['Embarked'] = test_df['Embarked'].map( {'C': 0, 'Q': 1, 'S':2} ).astype(int)



# preview the data

test_df.head()
# From pandas dataframe to numpy array

train_array = np.array(train_df)

test_array = np.array(test_df)



# Remove NaN and replace them by 0

where_are_NaNs = np.isnan(train_array)

train_array[where_are_NaNs] = 0

where_are_NaNs = np.isnan(test_array)

test_array[where_are_NaNs] = 0



print(train_array)
forest = RandomForestClassifier(n_estimators = 100)

forest.fit(train_array[::,2::], train_array[::,1])

predict = forest.predict(test_array[::,1::])

forest.score(train_array[::,2::], train_array[::,1])
submission = pd.DataFrame({

        "PassengerId": test_df["PassengerId"],

        "Survived": predict

    }).astype(int)

submission.to_csv('titanic.csv', index=False)