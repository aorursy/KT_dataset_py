# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import cross_val_score

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.svm import SVC

from sklearn.naive_bayes import GaussianNB

from xgboost import XGBClassifier

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#Training Data

train_df  = pd.read_csv("/kaggle/input/titanic/train.csv")

train_df.info()

#Test Data

test_df  = pd.read_csv("/kaggle/input/titanic/test.csv")

test_df.info()
#Identified the null values, fixing these features

#Starting with Age

train_df["Age"].isnull().sum()
test_df["Age"].isnull().sum()
#Age has 177, 86 null values in Train and Test data set respectively

#Fixing null values in Age using the mean

data = [train_df, test_df]

for dataset in data:

    mean = train_df["Age"].mean()

    dataset['Age'] = dataset['Age'].fillna(mean)

    dataset["Age"] = train_df["Age"].astype(float) 



for dataset in data:

    #dataset['Age'] = dataset['Age'].astype(int)

    dataset.loc[ dataset['Age'] <= 15, 'Age'] = 0

    dataset.loc[(dataset['Age'] > 15) & (dataset['Age'] <= 25), 'Age'] = 1

    dataset.loc[(dataset['Age'] > 25) & (dataset['Age'] <= 35), 'Age'] = 2

    dataset.loc[(dataset['Age'] > 35) & (dataset['Age'] <= 40), 'Age'] = 3

    dataset.loc[(dataset['Age'] > 40) & (dataset['Age'] <= 45), 'Age'] = 4

    dataset.loc[(dataset['Age'] > 45) & (dataset['Age'] <= 50), 'Age'] = 5

    #dataset.loc[(dataset['Age'] > 40) & (dataset['Age'] <= 66), 'Age'] = 6

    dataset.loc[ dataset['Age'] > 50, 'Age'] = 6



# let's see how it's distributed 

train_df['Age'].value_counts()
test_df.info()
train_df.info()
#Next feature to have the most null values is Cabin

train_df.Cabin.value_counts()

#The cabin seems to have a letter and number, let's try dropping the numbers, we'll have letters from A to G

deck = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7, "O": 8}

data = [train_df, test_df]

for dataset in data:

    dataset['CabinDeck'] = dataset['Cabin'].fillna("O")

    dataset['CabinDeck'] = dataset['Cabin'].astype(str).str[0] 

    #dataset['CabinDeck'] = dataset['CabinDeck'].str.capitalize()

    dataset['CabinDeck'] = dataset['CabinDeck'].map(deck)

    dataset['CabinDeck'] = dataset['CabinDeck'].fillna(0)

    dataset['CabinDeck'] = dataset['CabinDeck'].astype(int) 

#Dropping the Cabin feature. This can be done in two ways

#train_df = train_df.drop(['Cabin'], axis=1)

#test_df.drop(['Cabin'], axis=1, inplace = True)

train_df.head(10)
train_df = train_df.drop(['Cabin'], axis=1)

test_df.drop(['Cabin'], axis=1, inplace = True)
test_df['CabinDeck'].value_counts()
#Embarked is another feature which has a few null values

#Since it has very few missing values let's find the most common value for it and use that to fill the blanks

train_df.Embarked.value_counts()

#The most common value seems to be 'S'

common_value = 'S'

data = [train_df, test_df]

for dataset in data:

    dataset['Embarked'] = dataset['Embarked'].fillna(common_value)



train_df['Embarked'].describe()
train_df.info()

test_df.info()
#The Fare feature has just one missing value in the Test data set let's fix it

for dataset in data:

    dataset['Fare'] = dataset['Fare'].fillna(0)

#test_df.info()

for dataset in data:

    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0

    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1

    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2

    dataset.loc[(dataset['Fare'] > 31) & (dataset['Fare'] <= 99), 'Fare']   = 3

    dataset.loc[(dataset['Fare'] > 99) & (dataset['Fare'] <= 250), 'Fare']   = 4

    dataset.loc[ dataset['Fare'] > 250, 'Fare'] = 5

    dataset['Fare'] = dataset['Fare'].astype(int)



train_df['Fare'].value_counts()
#Now that all Null values are fixed let's convert the features that are of type Object into Integers

#Let's find the different values that these features have

train_df.Embarked.value_counts()
train_df.Sex.value_counts()
#Embarked feature has 3 unique values- S, C and Q, let's convert them into Integers

#Similarly Sex feature has 2 unique values- Male and Female, let's convert them as Integers too

embarkedMap = {"S": 0, "C": 1, "Q": 2}

genderMap = {"male": 0, "female": 1}



for dataset in data: 

    dataset['Embarked'] = dataset['Embarked'].map(embarkedMap)

    dataset['Sex'] = dataset['Sex'].map(genderMap)
train_df.info()
#Name doesn't seem to provide any insights to our analysis, let's see if we can extract a part of it and create a new feature. let's drop this feature

data = [train_df, test_df]

titles = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}

for dataset in data:

    dataset['Title'] = dataset['Name'].str.extract(' ([A-Za-z]+)\.')

    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr',\

                                            'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

    dataset['Title'] = dataset['Title'].map(titles)

    dataset['Title'] = dataset['Title'].fillna(0)

train_df = train_df.drop(['Name'], axis=1)

test_df = test_df.drop(['Name'], axis=1)

train_df['Title'].value_counts()


#Similarly, Ticket also doesn't seem to provide any insights, let's drop that column as well

train_df['Ticket'].value_counts()

train_df = train_df.drop(['Ticket'], axis=1)

test_df = test_df.drop(['Ticket'], axis=1) 
#Let's look at the data again and see if we need to make any other modifications

train_df.head(10)
#PassengerId also doesn't seem to be insightful, let's drop this column from our training dataset. 

#We do not want to remove this from our test data set since, this field will be used to submit our analysis

train_df = train_df.drop(['PassengerId'], axis=1)
#Let's begin with our analysis

#Logistic Regression Analysis

#Let's prepare our data for logistic regression, we need our features to be on the X axis and our predictable feature to be on the Y axis

X_train = train_df.drop('Survived', axis=1) #dropped the predictable feature from X axis

Y_train = train_df['Survived'] #added only predictable feature to Y axis
X_train.head(10)
#Removing the PassengerId feature from a copy of test data to aid with our analysis

X_test = test_df.drop("PassengerId", axis=1).copy()

X_test.head(10)
#Decision Tree

clf_dt = DecisionTreeClassifier(ccp_alpha=0.0, class_weight=None, criterion='entropy',

                       max_depth=None, max_features=None, max_leaf_nodes=None,

                       min_impurity_decrease=0.0, min_impurity_split=None,

                       min_samples_leaf=10, min_samples_split=35,

                       min_weight_fraction_leaf=0.0, presort='deprecated',

                       random_state=None, splitter='best')

clf_dt.fit(X_train, Y_train)

Y_pred  = clf_dt.predict(X_test)

scores = cross_val_score(clf_dt, X_train, Y_train, cv=10, scoring = "accuracy")

#Print the accuracy % of our model

print("Scores:", scores)

print("Mean:", scores.mean())

print("Standard Deviation:", scores.std())
'''#Random Forest



clf_rf = RandomForestClassifier(n_estimators=500)

scores = cross_val_score(clf_rf, X_train, Y_train, cv=10, scoring = "accuracy")



print("Scores:", scores)

print("Mean:", scores.mean())

print("Standard Deviation:", scores.std())'''
'''#SVM

clf_svm = SVC()

clf_svm.fit(X_train, Y_train) 

Y_pred  = clf_svm.predict(X_test)

scores = cross_val_score(clf_svm, X_train, Y_train, cv=10, scoring = "accuracy")

#Print the accuracy % of our model

print("Scores:", scores)

print("Mean:", scores.mean())

print("Standard Deviation:", scores.std())'''
'''#Gaussian Naive bayes

gnb = GaussianNB()

y_pred = gnb.fit(X_train, Y_train).predict(X_test)

scores = cross_val_score(clf_dt, X_train, Y_train, cv=10, scoring = "accuracy")

#Print the accuracy % of our model

print("Scores:", scores)

print("Mean:", scores.mean())

print("Standard Deviation:", scores.std())'''
#Bagging Classifier
'''#XGBoost Classifier

clf_xgb = XGBClassifier().fit(X_train, Y_train)

Y_pred  = clf_xgb.predict(X_test)

clf_xgb.score(X_train, Y_train)

acc_random_forest = round(clf_xgb.score(X_train, Y_train)*100, 2)

#scores = cross_val_score(clf, X_train, Y_train, cv=10, scoring = "accuracy")

#Print the accuracy % of our model

print("Scores:", scores)

print("Mean:", scores.mean())

print("Standard Deviation:", scores.std())'''
#Gradient Boosting Classifier
'''#Implementing Logic Regression as our Machine Learning algorithm



clf = LogisticRegression(random_state=0)

clf.fit(X_train, Y_train) #Fit the X_train and Y_train datasets in our classifier

Y_pred  = clf.predict(X_test) #Y_pred is our output from our test dataset

#clf.score(X_train, Y_train) #Calculate the score

#acc_logistic_reg = round(clf.score(X_train, Y_train)*100, 2) #Calculate the accuracy percentage of our model

scores = cross_val_score(clf, X_train, Y_train, cv=10, scoring = "accuracy")

#Print the accuracy % of our model



print("Scores:", scores)

print("Mean:", scores.mean())

print("Standard Deviation:", scores.std())'''



'''

Scores: [0.81111111 0.79775281 0.76404494 0.85393258 0.79775281 0.76404494

 0.80898876 0.83146067 0.83146067 0.83146067]

Mean: 0.8092009987515606

Standard Deviation: 0.027984546547837514

Scores: [0.81111111 0.82022472 0.76404494 0.86516854 0.82022472 0.7752809

 0.82022472 0.82022472 0.84269663 0.83146067]

Mean: 0.8170661672908863

Standard Deviation: 0.027947429020814398'''
#Implementing K Nearest Neighbors as our Machine Learning algorithm

'''

neigh = KNeighborsClassifier(n_neighbors=3)

neigh.fit(X_train, Y_train)

Y_pred  = neigh.predict(X_test)

neigh.score(X_train, Y_train) #Calculate the score

acc_nearest_neigh = round(neigh.score(X_train, Y_train)*100, 2) #Calculate the accuracy percentage of our model

#Print the accuracy % of our model

print ("Score:", acc_nearest_neigh)



#scores = cross_val_score(neigh, X_train, Y_train, cv=10, scoring = "accuracy")

#Print the accuracy % of our model

#print("Scores:", scores)

#print("Mean:", scores.mean())

#print("Standard Deviation:", scores.std())'''

'''

Scores: [0.63333333 0.68539326 0.74157303 0.74157303 0.78651685 0.74157303

 0.7752809  0.71910112 0.64044944 0.76404494]

Mean: 0.7228838951310862

Standard Deviation: 0.05076542434290519'''
#Let's print our prediction to a csv format with the PassengerId and our prediction whether they survived or not

output = pd.DataFrame({'PassengerId': test_df.PassengerId, 'Survived': Y_pred})

output.to_csv('my_submission.csv', index=False)

print("Your submission was successfully saved!")