# linear algebra

import numpy as np 



# data processing

import pandas as pd 



# data visualization

import seaborn as sns

%matplotlib inline

from matplotlib import pyplot as plt

from matplotlib import style



# Training Algorithms

from sklearn.model_selection import StratifiedShuffleSplit

from sklearn.metrics import accuracy_score, log_loss

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis

from sklearn.linear_model import LogisticRegression



test_df = pd.read_csv('../input/test.csv')

train_df = pd.read_csv('../input/train.csv')
train_df.info()





print ('-'*50)

test_df .info()
train_df.head(5)
train_df.describe(include=['O'])
train_df = train_df.drop(['Name','Ticket','Cabin'],axis=1)

test_df= test_df.drop(['Name','Ticket','Cabin'],axis=1)

train_df.head()
train_df = train_df.drop(['PassengerId'],axis=1)

#test_df= test_df.drop(['PassengerId'],axis=1)

train_df.head()
#Look at survival rate by Pclass

train_df.groupby('Pclass')[['Survived']].mean()
#Look at survival rate by Embarked

train_df.groupby('Embarked')[['Survived']].mean()
#Look at survival rate by Sex

train_df.groupby('Sex')[['Survived']].mean()
#Look at survival rate by Pclass

train_df.groupby('SibSp')[['Survived']].mean()
#Look at survival rate by Pclass

train_df.groupby('Parch')[['Survived']].mean()
data = [train_df, test_df]

for dataset in data:

    mean = train_df["Age"].mean()

    std = test_df["Age"].std()

    is_null = dataset["Age"].isnull().sum()

    # compute random numbers between the mean, std and is_null

    

    rand_age = np.random.randint(mean - std, mean + std, size = is_null)

    # fill NaN values in Age column with random values generated

    age_slice = dataset["Age"].copy()

    age_slice[np.isnan(age_slice)] = rand_age

    dataset["Age"] = age_slice

    dataset["Age"] = train_df["Age"].astype(int)

train_df["Age"].isnull().sum()
train_df['Embarked'].describe()
most_represented = 'S'



for dataset in data:

    dataset['Embarked'] = dataset['Embarked'].fillna(most_represented)
train_df.info()
#Sex

genders = {"male": 0, "female": 1}

data = [train_df, test_df]



for dataset in data:

    dataset['Sex'] = dataset['Sex'].map(genders)
#Embarked

for dataset in data:

    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)



train_df.head()
train_df['AgeRange'] = pd.cut(train_df['Age'], 6)

train_df[['AgeRange', 'Survived']].groupby(['AgeRange'], as_index=False).mean().sort_values(by='AgeRange', ascending=True)
data = [train_df, test_df]

for dataset in data:

    dataset['Age'] = dataset['Age'].astype(int)

    dataset.loc[ dataset['Age'] <= 13.33, 'Age'] = 0

    dataset.loc[(dataset['Age'] > 13.33) & (dataset['Age'] <= 26.66), 'Age'] = 1

    dataset.loc[(dataset['Age'] > 26.66) & (dataset['Age'] <= 40), 'Age'] = 2

    dataset.loc[(dataset['Age'] > 40) & (dataset['Age'] <=53.33), 'Age'] = 3

    dataset.loc[(dataset['Age'] > 53.33) & (dataset['Age'] <= 66.66), 'Age'] = 4

  

    dataset.loc[ dataset['Age'] > 66, 'Age'] = 5

    



train_df.head()
# we can now drop Ange Range

train_df = train_df.drop(['AgeRange'], axis=1)
test_df['Fare'].fillna(test_df['Fare'].dropna().median(), inplace=True)

test_df.head()
#Fare

train_df['FareRange'] = pd.qcut(train_df['Fare'], 4)

train_df[['FareRange', 'Survived']].groupby(['FareRange'], as_index=False).mean().sort_values(by='FareRange', ascending=True)
data = [train_df, test_df]

for dataset in data:

    

    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0

    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1

    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2

    dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3

    dataset['Fare'] = dataset['Fare'].astype(int)

train_df = train_df.drop(['FareRange'], axis=1)



    

train_df.head(10)
data = [train_df, test_df]

for dataset in data:

    dataset['Family'] = dataset['SibSp'] + dataset['Parch'] + 1



train_df[['Family', 'Survived']].groupby(['Family'], as_index=False).mean().sort_values(by='Survived', ascending=False)
for dataset in data:

    dataset['Single'] = 0

    dataset.loc[dataset['Family'] == 1, 'Single'] = 1



train_df[['Single', 'Survived']].groupby(['Single'], as_index=False).mean()
test_df = test_df.drop(['Parch', 'SibSp', 'Family'], axis=1)



train_df=train_df.drop(['Parch', 'SibSp', 'Family'], axis=1)



train_df.head()

test_df.head()
X_train = train_df.drop("Survived", axis=1)

Y_train = train_df["Survived"]

X_test  = test_df.drop("PassengerId", axis=1).copy()

X_train.shape, Y_train.shape, X_test.shape
classifiers = [

    KNeighborsClassifier(),

    SVC(),

    DecisionTreeClassifier(),

    RandomForestClassifier(),

    AdaBoostClassifier(),

    GradientBoostingClassifier(),

    GaussianNB(),

    LinearDiscriminantAnalysis(),

    QuadraticDiscriminantAnalysis(),

    LogisticRegression()]

for clf in classifiers:

    clf.fit(X_train, Y_train)

    clf_Pred=clf.predict(X_test)

    acc_clf = round(clf.score(X_train, Y_train) * 100, 2)

    print(clf.__class__.__name__,acc_clf)