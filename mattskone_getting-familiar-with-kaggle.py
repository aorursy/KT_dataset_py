import numpy as np

import pandas as pd

from sklearn import svm

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import cross_val_score

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB
train_df = pd.read_csv('../input/train.csv')

test_df = pd.read_csv('../input/test.csv')
# Data Exploration Cell



#train_df.info()

#train_df.Embarked.value_counts()

#dir(train_df.Embarked)

#test_df.Fare.isnull().sum()

#train_df.Family.value_counts()

#test_df.Fare.loc[test_df.Embarked.isnull() == True].count()

#train_df.count()

#test_df.Fare.count()
# Drop columns with no analytical value (either irrelevant or too sparse)

train_df.drop(['PassengerId', 'Cabin', 'Name','Ticket'], axis=1, inplace=True)

test_df.drop(['Cabin', 'Name','Ticket'], axis=1, inplace=True)
# EMBARKED



# Fill empty values for Embarked with most frequent value:

train_df['Embarked'].fillna('S', inplace=True)



# Convert to numeric values

train_df.loc[train_df['Embarked'] == 'S', 'Embarked'] = 0

train_df.loc[train_df['Embarked'] == 'C', 'Embarked'] = 1

train_df.loc[train_df['Embarked'] == 'Q', 'Embarked'] = 2



test_df.loc[test_df['Embarked'] == 'S', 'Embarked'] = 0

test_df.loc[test_df['Embarked'] == 'C', 'Embarked'] = 1

test_df.loc[test_df['Embarked'] == 'Q', 'Embarked'] = 2
# FARE



# Fill empty values for Fare with median fare:

test_df['Fare'].fillna(test_df['Fare'].median(), inplace=True)
# AGE



# Fill empty values from a gaussian of the known values

train_age_fill = np.random.normal(loc=train_df.Age.mean(),

                                  scale=train_df.Age.std(),

                                  size=train_df.Age.isnull().sum())

train_df.loc[np.isnan(train_df['Age']), 'Age'] = train_age_fill



test_age_fill = np.random.normal(loc=test_df.Age.mean(),

                                  scale=test_df.Age.std(),

                                  size=test_df.Age.isnull().sum())

test_df.loc[np.isnan(test_df['Age']), 'Age'] = test_age_fill
# SIBSP and PARCH



train_df['Family'] = train_df['Parch'] + train_df['SibSp']

train_df.loc[train_df['Family'] > 0, 'Family'] = 1

train_df.loc[train_df['Family'] == 0, 'Family'] = 0



test_df['Family'] =  test_df['Parch'] + test_df['SibSp']

test_df.loc[test_df['Family'] > 0, 'Family'] = 1

test_df.loc[test_df['Family'] == 0, 'Family'] = 0



train_df.drop(['SibSp','Parch'], axis=1, inplace=True)

test_df.drop(['SibSp','Parch'], axis=1, inplace=True)
# SEX



# Convert to numeric values

train_df.loc[train_df['Sex'] == 'male', 'Sex'] = 0

train_df.loc[train_df['Sex'] == 'female', 'Sex'] = 1



test_df.loc[test_df['Sex'] == 'male', 'Sex'] = 0

test_df.loc[test_df['Sex'] == 'female', 'Sex'] = 1
X_train = train_df.drop('Survived', axis=1)

Y_train = train_df['Survived']

X_test = test_df.drop('PassengerId', axis=1)
#classifier = RandomForestClassifier(n_estimators=100)  # 0.8 +/- 0.04

classifier = svm.SVC(kernel='poly')  # 0.79 +/- 0.04

#classifier = GaussianNB()  # 0.79 +/- 0.02

#classifier = KNeighborsClassifier(n_neighbors=10)  # 0.68 +/- 0.08

#classifier = svm.LinearSVC(penalty='l1', dual=False)  # 0.79 +/- 0.04

scores = cross_val_score(classifier, X_train, Y_train, cv=5)

print('Accuracy: %0.2f (+/- %0.2f)' % (scores.mean(), scores.std() * 2))



#classifier.fit(X_train, Y_train)

#Y_pred = classifier.predict(X_test)
#submission = pd.DataFrame({

#        "PassengerId": test_df["PassengerId"],

#        "Survived": Y_pred

#    })

#submission.to_csv('titanic.csv', index=False)