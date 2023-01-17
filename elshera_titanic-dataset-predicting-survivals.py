# Load the neccessary Libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns
df = pd.read_csv('../input/train.csv')
# Explore the dataset to see values, datatypes, nan values

df.info()
# data dimensions

df.shape
# show some initial rows 

df.head(10)
# For the Cathegorical variables we should see the set of their values over all samples.

print('Pclass:   ', df['Pclass'].unique())

print('Parch:    ', df['Parch'].unique())

print('SibSp:    ', df['SibSp'].unique())

print('Embarked: ', df['Embarked'].unique())



# Let see if the 209 cabines in the dataset host each one person or more

print('Cabin:    ', df['Cabin'].unique().size)



# Are the Ticket unique

print('Ticket:   ', df['Ticket'].unique().size)
# It makes sense to extract those columns which contain booleant values like 0-1 or Trus - False. 

colist = df.columns.tolist()

for col in colist:

    if df[col].unique().size == 2:

        print(col) 
# Embarked have 2 missing values. With low error we can fill them based on the most likely value.

print(df[df['Embarked'] == 'S']['Embarked'].size)

print(df[df['Embarked'] == 'Q']['Embarked'].size)

print(df[df['Embarked'] == 'C']['Embarked'].size)
# The most recurrent value is "S" therefore we can fill the missing values in Embarked with S and replace labels with 

# numbers

df["Embarked"] = df["Embarked"].fillna('S')

df['Embarked'] = df['Embarked'].map({'S': 2, 'C': 1, 'Q':0})
# Could replace SibSp and Parch with one column, called Family, indicating the overall family number. 

# I do not see much difference if one is a brother or a sister from beeing a mother or a father.

df['Family'] = df['SibSp'] + df['Parch'] + 1
# Consider now replacing the Sex type from a label to a number.

df['Sex'] = df['Sex'].map({'male': 1, 'female': 0})
# Consider the distribution of the age

age_mean = df['Age'].mean()

age_std  = df['Age'].std()



# generate random numbers age_mean

rand_age = np.random.randint(age_mean - age_std, age_mean + age_std, size = df["Age"].isnull().sum())



# fill in the randome values to the missing age values

df["Age"][np.isnan(df["Age"])] = rand_age



# convert from float to int

df['Age'] = df['Age'].astype(int)



# There are some columns which can be dropped already

df.drop(['Parch', 'SibSp', 'PassengerId'], axis=1 , inplace=True)



# finally look at the dataset

df.info()

df.head(6)
df.drop(['Cabin', 'Ticket', 'Name'], axis=1 , inplace=True)

df.info()

df.head(6)
df.describe()
# To better consider correlation between the various fields, we might use a correlation matrix.

correlation_matrix = df.corr()

plt.figure(figsize=(10,10))

sns.heatmap(correlation_matrix, vmax=1, square=True, annot=True, cmap='Reds')

plt.title('Overall Correlation Matrix')

plt.show()
df.drop(['Fare'], axis=1, inplace=True)

df.head(4)
# collect all the survival

Survive = df[df['Survived'] == 1]



# create groups based on them

by_families = Survive.groupby(df['Family'])

by_age = Survive.groupby(df['Age'])

by_embark = Survive.groupby('Embarked')

by_sex = Survive.groupby('Sex')

by_class = Survive.groupby('Pclass')



## See how the groups relates with Survived

#print(df.groupby(df['Family'])['Survived'].count())

print(by_families['Survived'].count() / df.groupby(df['Family'])['Survived'].count() * 100 * (df.groupby(df['Family'])['Survived'].count()/891))

print(by_embark['Survived'].count() / df.groupby(df['Embarked'])['Survived'].count() * 100 * (df.groupby(df['Embarked'])['Survived'].count()/891))

print(by_sex['Survived'].count() / df.groupby(df['Sex'])['Survived'].count() * 100 * (df.groupby(df['Sex'])['Survived'].count()/891))

print(by_class['Survived'].count() / df.groupby(df['Pclass'])['Survived'].count() * 100 * (df.groupby(df['Pclass'])['Survived'].count()/891))

#print(by_age['Survived'].count() / df.groupby(df['Age'])['Survived'].count() * 100 *(df.groupby(df['Age'])['Survived'].count()/891))

age_distribution = by_age['Survived'].count() / df.groupby(df['Age'])['Survived'].count() * 100 * (df.groupby(df['Age'])['Survived'].count()/891)

age_distribution.hist()
# Prepare the data. One of the first things to do is to split between the target to be predicted Y, and the variable 

Y = df['Survived'].values

np.shape(Y)
# before creating the feature matriy we need to drop the target "Survived"

df.drop(['Survived'], axis=1 , inplace=True)

df.info()

df.head(4)
# create features matrix

X = df.as_matrix()

print(np.shape(X))



# show the second row

X[1]
# load and prepare the test data.

df_test = pd.read_csv('../input/test.csv')

df_test.info()
# We already know the data to be dropped

ID_test = df_test['PassengerId'].values

df_test.drop(['Cabin', 'Fare', 'Ticket', 'PassengerId', 'Name'], axis=1, inplace=True)

df_test.head(3)
# Filling missing data from the Age

age_mean = df_test['Age'].mean()

age_std  = df_test['Age'].std()

rand_age = np.random.randint(age_mean - age_std, age_mean + age_std, size = df_test["Age"].isnull().sum())

df_test["Age"][np.isnan(df_test["Age"])] = rand_age

df_test['Age'] = df_test['Age'].astype(int)



# Consider now replacing the Sex type from a label to a number.

df_test['Sex'] = df_test['Sex'].map({'male': 1, 'female': 0})



# Could replace SibSp and Parch with one column, called Family

df_test['Family'] = df_test['SibSp'] + df_test['Parch']



df_test['Embarked'] = df_test['Embarked'].map({'S': 2, 'C': 1, 'Q':0})



# We can drop now SibSp and Parch

df_test.drop(['Parch', 'SibSp'], axis=1 , inplace=True)



df_test.info()

df_test.head(4)
# create features matrix

X_test = df_test.as_matrix()

print(np.shape(X_test))
# Linear Regression

from sklearn.linear_model import LogisticRegression

logistic = LogisticRegression()

logistic.fit(X, Y)

Y_pred_lg = logistic.predict(X_test)

logistic.score(X, Y)
# Random Forests

from sklearn.ensemble import RandomForestClassifier

random_forest = RandomForestClassifier(n_estimators=100)

random_forest.fit(X, Y)

Y_pred_rf = random_forest.predict(X_test)

random_forest.score(X, Y)
# Support Vector Machine

from sklearn.svm import SVC

svc = SVC(C=1, kernel='rbf').fit(X, Y)

Y_pred_svc = svc.predict(X_test)

svc.score(X, Y)
from sklearn import tree

clf = tree.DecisionTreeClassifier().fit(X, Y)

Y_pred_clf = clf.predict(X_test)

clf.score(X, Y)
from sklearn.neighbors import KNeighborsClassifier

knb = KNeighborsClassifier(n_neighbors = 3)

knb.fit(X, Y)

Y_pred_knb = knb.predict(X_test)

knb.score(X, Y)
from sklearn.naive_bayes import GaussianNB

gaus = GaussianNB()

gaus.fit(X, Y)

Y_pred_gs = gaus.predict(X_test)

gaus.score(X, Y)
Y_pred = Y_pred_gs + Y_pred_knb + Y_pred_clf + Y_pred_svc + Y_pred_rf

type(Y_pred)
# If 3 or more out of 5 predictors vote for 1 than we have Survived = 1 otherwise we have Survived = 0

Y_pred[Y_pred < 2.5] = 0

Y_pred[Y_pred > 2.5] = 1

Y_pred
submission = pd.DataFrame({

        "PassengerId": ID_test,

        "Survived": Y_pred

    })

    

submission.to_csv("titanic_solution.csv", index=False)