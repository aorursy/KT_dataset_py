# Import Libraries

%matplotlib inline

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



import warnings

warnings.filterwarnings("ignore")
# Load Data

train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")



train.head()
print("-------------------Training Data Information--------------------")

train.info()

print("-------------------Test Data Information------------------------")

test.info()
# Removing irrelevant columns

# PassengerID --> irrelevant

# Name --> may categorize it, but I decided to drop

# Ticket --> Irrelevant

# Cabin --> Very few entries

train = train.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis = 1)

test = test.drop(['Name', 'Ticket', 'Cabin'], axis = 1)



list(train.columns.values)
# Analyzing each attribute

# Pclass



fig, (axis1, axis2) = plt.subplots(1, 2, figsize = (10, 5))



# plot1

sns.countplot('Pclass', hue = 'Survived', data = train, ax = axis1)

# plot2

class_per = train[['Pclass', 'Survived']].groupby(['Pclass'], as_index = False).mean()

sns.barplot('Pclass', 'Survived', data = class_per, ax = axis2)



# create dummy variables for training data

train_class_dummies = pd.get_dummies(train['Pclass'])

train_class_dummies.columns = ['Upper', 'Middle', 'Lower']

train_class_dummies.drop(['Lower'], axis = 1, inplace = True)



# create dummy variables for test data

test_class_dummies = pd.get_dummies(test['Pclass'])

test_class_dummies.columns = ['Upper', 'Middle', 'Lower']

test_class_dummies.drop(['Lower'], axis = 1, inplace = True)



# drop Pclass column from training and test data

train.drop(['Pclass'], axis = 1, inplace = True)

test.drop(['Pclass'], axis = 1, inplace = True)



# join dummy variables

train = train.join(train_class_dummies)

test = test.join(test_class_dummies)
#Sex



fig, (axis1, axis2) = plt.subplots(1, 2, figsize = (10, 5))



# plot1

sns.countplot('Sex', hue = 'Survived', data = train, ax = axis1)

# plot2

sns.barplot('Sex', 'Survived', data = train, ax = axis2)



# create dummy variables for training data

train_sex_dummies = pd.get_dummies(train['Sex'])

train_sex_dummies.drop(['male'], axis = 1, inplace = True)



# create dummy variables for test data

test_sex_dummies = pd.get_dummies(test['Sex'])

test_sex_dummies.drop(['male'], axis = 1, inplace = True)



# drop Sex column from training and test data

train.drop(['Sex'], axis = 1, inplace = True)

test.drop(['Sex'], axis = 1, inplace = True)



# join dummy variables

train = train.join(train_sex_dummies)

test = test.join(test_sex_dummies)
#Age



# mean age of training and test data

train_mean_age = train['Age'].mean()

test_mean_age = test['Age'].mean()



# std age of training and test data

train_std_age = train['Age'].std()

test_std_age = test['Age'].std()



# count of missing values

train_missing_count_age = train['Age'].isnull().sum()

test_missing_count_age = test['Age'].isnull().sum()



# random values for training and test data

train_random_age = np.random.randint(train_mean_age - train_std_age, train_mean_age + train_std_age,

                                    size = train_missing_count_age)

test_random_age = np.random.randint(test_mean_age - test_std_age, test_mean_age + test_std_age,

                                    size = test_missing_count_age)



train.loc[train.Age.isnull(), 'Age'] = train_random_age

test.loc[test.Age.isnull(), 'Age'] = test_random_age



# categorize age column in training data

train['Age'] = train['Age'].astype(int)

bins = (0, 5, 12, 18, 25, 35, 60, 120)

group_names = ['Baby', 'Child', 'Teenager', 'Student', 'Young Adult', 'Adult', 'Senior']

train['Age'] = pd.cut(train['Age'], bins, labels=group_names, right=True, include_lowest=True)



# catgorize age column in test data

test['Age'] = test['Age'].astype(int)

test['Age'] = pd.cut(test['Age'], bins, labels=group_names, right=True, include_lowest=True)



fig, (axis1, axis2, axis3) = plt.subplots(1, 3, figsize = (15, 5))



# plot1

sns.countplot('Age', hue = 'Survived', data = train, ax = axis1)

# plot2

class_per = train[['Age', 'Survived']].groupby(['Age'], as_index = False).mean()

sns.barplot('Age', 'Survived', data = class_per, ax = axis2)

# plot3

sns.barplot(x="Age", y="Survived", hue="female", data=train, ax = axis3);



# create dummy variables for training data

train_age_dummies = pd.get_dummies(train['Age'])

train_age_dummies.drop(['Child'], axis = 1, inplace = True)



# create dummy variables for test data

test_age_dummies = pd.get_dummies(test['Age'])

test_age_dummies.drop(['Child'], axis = 1, inplace = True)



# drop Age column from training and test data

train.drop(['Age'], axis = 1, inplace = True)

test.drop(['Age'], axis = 1, inplace = True)



# join dummy variables

train = train.join(train_age_dummies)

test = test.join(test_age_dummies)
# SibSp and Parch

# replace these columns with a new family column



train['Family'] = train['SibSp'] + train['Parch']

train['Family'].loc[train['Family'] > 0] = 1

train['Family'].loc[train['Family'] == 0] = 0



test['Family'] = test['SibSp'] + test['Parch']

test['Family'].loc[train['Family'] > 0] = 1

test['Family'].loc[train['Family'] == 0] = 0



# drop SibSp and Parch

train = train.drop(['SibSp', 'Parch'], axis = 1)

test = test.drop(['SibSp', 'Parch'], axis = 1)



fig, (axis1, axis2) = plt.subplots(1, 2, figsize = (10, 5))

#plot1

sns.countplot('Family', data = train, ax = axis1)

sns.barplot('Family', 'Survived', data = train, ax = axis2)

# Embarked



# fill missing values

train['Embarked'] = train['Embarked'].fillna('S')



fig, (axis1, axis2, axis3) = plt.subplots(1, 3, figsize = (15, 5))

#plot1

sns.countplot('Embarked', data = train, ax = axis1)

#plot2

sns.countplot('Survived', hue = 'Embarked', data = train, ax = axis2)

#plot3

sns.barplot('Embarked', 'Survived', data = train, ax = axis3)



# create dummy variables for training data

train_embark_dummies = pd.get_dummies(train['Embarked'])

train_embark_dummies.drop(['S'], axis = 1, inplace = True)



# create dummy variables for test data

test_embark_dummies = pd.get_dummies(test['Embarked'])

test_embark_dummies.drop(['S'], axis = 1, inplace = True)



# drop Embarked column from training and test data

train.drop(['Embarked'], axis = 1, inplace = True)

test.drop(['Embarked'], axis = 1, inplace = True)



# join dummy variables

train = train.join(train_embark_dummies)

test = test.join(test_embark_dummies)

#Fare



# fill missing values

test["Fare"].fillna(test["Fare"].median(), inplace=True)



# catgorize fare column in training data

train['Fare'] = train['Fare'].astype(int)

bins = (0, 8, 15, 30, 1000)

group_names = ['Less_Fare', 'Average_Fare', 'Med_Fare', 'More_Fare']

train['Fare'] = pd.cut(train['Fare'], bins, labels=group_names, right=True, include_lowest=True)



# catgorize fare column in test data

test['Fare'] = test['Fare'].astype(int)

test['Fare'] = pd.cut(test['Fare'], bins, labels=group_names, right=True, include_lowest=True)



fig, (axis1, axis2) = plt.subplots(1, 2, figsize = (10, 5))



# plot1

sns.countplot('Fare', hue = 'Survived', data = train, ax = axis1)

# plot2

fare_per = train[['Fare', 'Survived']].groupby(['Fare'], as_index = False).mean()

sns.barplot('Fare', 'Survived', data = fare_per, ax = axis2)



# create dummy variables for training data

train_fare_dummies = pd.get_dummies(train['Fare'])

train_fare_dummies.drop(['Less_Fare'], axis = 1, inplace = True)



# create dummy variables for test data

test_fare_dummies = pd.get_dummies(test['Fare'])

test_fare_dummies.drop(['Less_Fare'], axis = 1, inplace = True)



# drop Age column from training and test data

train.drop(['Fare'], axis = 1, inplace = True)

test.drop(['Fare'], axis = 1, inplace = True)



# join dummy variables

train = train.join(train_fare_dummies)

test = test.join(test_fare_dummies)

# Create test and train sets



X_train = train.drop(['Survived'], axis = 1)

Y_train = train['Survived']

X_test = test.drop(['PassengerId'], axis = 1).copy()
# K nearest neighbours

from sklearn.neighbors import KNeighborsClassifier



knn = KNeighborsClassifier(n_neighbors = 3)

knn.fit(X_train, Y_train)

Y_pred = knn.predict(X_test)

knn.score(X_train, Y_train)
# Support Vector Machines

from sklearn.svm import SVC, LinearSVC



svc = SVC()

svc.fit(X_train, Y_train)

Y_pred = svc.predict(X_test)

svc.score(X_train, Y_train)
# Gaussian Naive Bayes

from sklearn.naive_bayes import GaussianNB



gaussian = GaussianNB()

gaussian.fit(X_train, Y_train)

Y_pred = gaussian.predict(X_test)

gaussian.score(X_train, Y_train)
# Random Forests

from sklearn.ensemble import RandomForestClassifier



random_forest = RandomForestClassifier(n_estimators=100)

random_forest.fit(X_train, Y_train)

Y_pred = random_forest.predict(X_test)

random_forest.score(X_train, Y_train)
# Logistic Regression

from sklearn.linear_model import LogisticRegression



logreg = LogisticRegression()

logreg.fit(X_train, Y_train)

Y_pred = logreg.predict(X_test)

logreg.score(X_train, Y_train)
# get Correlation Coefficient for each feature using Logistic Regression

coeff_df = pd.DataFrame(train.columns.delete(0))

coeff_df.columns = ['Features']

coeff_df["Coefficient Estimate"] = pd.Series(logreg.coef_[0])



# preview

coeff_df
submission = pd.DataFrame({

        "PassengerId": test["PassengerId"],

        "Survived": Y_pred

    })

submission.to_csv('titanic.csv', index=False)