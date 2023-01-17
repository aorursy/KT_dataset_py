import pandas as pd

import numpy as np

train_df = pd.read_csv('../input/train.csv')

test_df = pd.read_csv('../input/test.csv')
train_df.info()
train_df['Embarked'] = train_df['Embarked'].fillna('S')

train_X = train_df[['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']]

test_X = test_df[['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']]
train_X.is_copy = False

test_X.is_copy = False

d = {'S':0, 'C':1, 'Q':2}

train_X['Embarked'] = train_df['Embarked'].map(lambda x: d[x])

test_X['Embarked'] = test_df['Embarked'].map(lambda x: d[x])
# get average, std, and number of NaN values in titanic_df

average_age_titanic   = train_X["Age"].mean()

std_age_titanic       = train_X["Age"].std()

count_nan_age_titanic = train_X["Age"].isnull().sum()

# get average, std, and number of NaN values in test_df

average_age_test   = test_X["Age"].mean()

std_age_test       = test_X["Age"].std()

count_nan_age_test = test_X["Age"].isnull().sum()

# generate random numbers between (mean - std) & (mean + std)

rand_1 = np.random.randint(average_age_titanic - std_age_titanic, average_age_titanic + std_age_titanic, size = count_nan_age_titanic)

rand_2 = np.random.randint(average_age_test - std_age_test, average_age_test + std_age_test, size = count_nan_age_test)

                           

# fill NaN values in Age column with random values generated

train_X['Age'][np.isnan(train_X['Age'])] = rand_1

test_X['Age'][np.isnan(test_X['Age'])] = rand_2
train_X['Sex'] = train_df['Sex'].map(lambda x:{'male':0,'female':1}[x]).astype(int)

test_X['Sex'] = test_df['Sex'].astype(int)

train_Y = train_df['Survived'].astype(int)

















submission = pd.DataFrame({

        "PassengerId": test_df["PassengerId"],

        "Survived": Y_pred

    })

submission.to_csv('titanic.csv', index=False)
import pandas as pd

import numpy as np

train_df = pd.read_csv('../input/train.csv')

test_df = pd.read_csv('../input/test.csv')



train_df['Embarked'] = train_df['Embarked'].fillna('S')

train_X = train_df[['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']]

test_X = test_df[['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']]





train_X.is_copy = False

test_X.is_copy = False

d = {'S':0, 'C':1, 'Q':2}

train_X['Embarked'] = train_df['Embarked'].map(lambda x: d[x])

test_X['Embarked'] = test_df['Embarked'].map(lambda x: d[x])





# get average, std, and number of NaN values in titanic_df

average_age_titanic   = train_X["Age"].mean()

std_age_titanic       = train_X["Age"].std()

count_nan_age_titanic = train_X["Age"].isnull().sum()

# get average, std, and number of NaN values in test_df

average_age_test   = test_X["Age"].mean()

std_age_test       = test_X["Age"].std()

count_nan_age_test = test_X["Age"].isnull().sum()

# generate random numbers between (mean - std) & (mean + std)

rand_1 = np.random.randint(average_age_titanic - std_age_titanic, average_age_titanic + std_age_titanic, size = count_nan_age_titanic)

rand_2 = np.random.randint(average_age_test - std_age_test, average_age_test + std_age_test, size = count_nan_age_test)

                           

# fill NaN values in Age column with random values generated

train_X['Age'][np.isnan(train_X['Age'])] = rand_1

test_X['Age'][np.isnan(test_X['Age'])] = rand_2



d = {'male':0,'female':1}

train_X['Sex'] = train_df['Sex'].map(lambda x:d[x]).astype(int)

test_X['Sex'] = test_df['Sex'].map(lambda x:d[x]).astype(int)





train_Y = train_df['Survived'].astype(int)



# machine learning

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB





# Random Forests



random_forest = RandomForestClassifier(n_estimators=100)



random_forest.fit(train_X, train_Y)





test_X.Fare = test_X.Fare.fillna(test_X.Fare.mean())



Y_pred = random_forest.predict(test_X)



random_forest.score(train_X, train_Y)
submission = pd.DataFrame({

        "PassengerId": test_df["PassengerId"],

        "Survived": Y_pred

    })

submission.to_csv('titanic.csv', index=False)