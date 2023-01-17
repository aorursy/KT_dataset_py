# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.ensemble import RandomForestClassifier



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.



titanic_df = pd.read_csv("../input/train.csv")

test_df    = pd.read_csv("../input/test.csv")
titanic_df.info()

print("----------------------------")

test_df.info()
titanic_df.describe()
#male = 0, female = 1



titanic_df['Sex'] = titanic_df['Sex'].replace('male', 0)

titanic_df['Sex'] = titanic_df['Sex'].replace('female', 1)

test_df['Sex'] = titanic_df['Sex'].replace('male', 0)

test_df['Sex'] = titanic_df['Sex'].replace('female', 1)

titanic_df[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)
freq_port = titanic_df.Embarked.dropna().mode()[0]

titanic_df['Embarked'] = titanic_df['Embarked'].fillna(freq_port)

test_df['Embarked'] = test_df['Embarked'].fillna(freq_port)

test_df.info()
titanic_df['Embarked'] = titanic_df['Embarked'].map( {'Q': 2 ,'C': 1, 'S': 0} ).astype(int)

test_df['Embarked'] = test_df['Embarked'].map( {'Q': 2 ,'C': 1, 'S': 0} ).astype(int)
titanic_df['Age'] = titanic_df['Age'].fillna(29.699118)

test_df['Age'] = test_df['Age'].fillna(29.699118)

test_df['Fare'] = test_df['Fare'].fillna(32.204208)
number_of_age_categories = 5

age_max = titanic_df.Age.max()

age_min = titanic_df.Age.min()

titanic_df.Age = titanic_df.Age.apply(lambda x: x/number_of_age_categories)
#preprocess data before training the model



titanic_results = titanic_df[['Survived']]

titanic_df = titanic_df.drop(['Survived'], 1)



#Discarding irrelevant features



droped_features = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'SibSp', 'Parch']



titanic_df = titanic_df.drop(droped_features, 1)

test_df = test_df.drop(droped_features, 1)
random_forest = RandomForestClassifier(n_estimators=100)

random_forest.fit(titanic_df, titanic_results)

Y_pred = random_forest.predict(test_df)

random_forest.score(titanic_df, titanic_results)

acc_random_forest = round(random_forest.score(titanic_df, titanic_results) * 100, 2)

acc_random_forest

IDs = np.arange(892, 1310)

values = np.column_stack((IDs, Y_pred))

temp = np.array([["PassengerId", "Survived"]])

values = np.concatenate((temp, values), axis = 0)



df = pd.DataFrame(values)
df.to_csv("output.csv", index = False)