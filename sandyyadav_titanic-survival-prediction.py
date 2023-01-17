# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style('whitegrid')

%matplotlib inline



#Import ML Algos

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



#from subprocess import check_output

#print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.



#Load data

train_df = pd.read_csv("../input/train.csv")

test_df = pd.read_csv("../input/test.csv")
#____________________________ AGE___________________________________#

#Check and fill age for training data

#fig, (axis1, axis2) = plt.subplots(1, 2, sharex = True, figsize = (15,5))

#axis1.set_title('Original Age')

#train_df['Age'].dropna().astype(int).hist(bins=70, ax=axis1)



average_age_train   = train_df["Age"].mean()

std_age_train      = train_df["Age"].std()

count_nan_age_train = train_df["Age"].isnull().sum()



train_rand = np.random.randint(average_age_train - std_age_train, average_age_train + std_age_train, size = count_nan_age_train)

train_df["Age"][np.isnan(train_df["Age"])] = train_rand



#axis2.set_title('Coverted Age')

#train_df['Age'].hist(bins=70, ax=axis2)



#Check and fill age for test data

#fig, (axis1, axis2) = plt.subplots(1, 2, sharex = True, figsize = (15,5))

#axis1.set_title('Original Age')

#test_df['Age'].dropna().astype(int).hist(bins=70, ax=axis1)



average_age_test   = test_df["Age"].mean()

std_age_test      = test_df["Age"].std()

count_nan_age_test = test_df["Age"].isnull().sum()



test_rand = np.random.randint(average_age_test - std_age_test, average_age_test + std_age_test, size = count_nan_age_test)

test_df["Age"][np.isnan(test_df["Age"])] = test_rand



#axis2.set_title('Coverted Age')

#test_df['Age'].hist(bins=70, ax=axis2)



train_df['Age'] = train_df['Age'].astype(int)

test_df['Age']    = test_df['Age'].astype(int)
#____________________________SEX_______________________________#

train_df['Sex'].loc[train_df['Sex'] == 'female'] = 0

test_df['Sex'].loc[test_df['Sex'] == 'female'] = 0



train_df['Sex'].loc[train_df['Sex'] == 'male'] = 1

test_df['Sex'].loc[test_df['Sex'] == 'male'] = 1



train_df['Sex'].loc[train_df['Age'] < 16] = 2

test_df['Sex'].loc[test_df['Age'] < 16] = 2



#train_df['Survived'].groupby(train_df['Sex']).mean().unstack()

train_df['Survived'].groupby([train_df['Sex'], train_df['Pclass']]).mean().unstack()





#___________________________Family_____________________________#

train_df['Family'] =  train_df["Parch"] + train_df["SibSp"]

train_df['Family'].loc[train_df['Family'] > 0] = 1

train_df['Family'].loc[train_df['Family'] == 0] = 0



test_df['Family'] =  test_df["Parch"] + test_df["SibSp"]

test_df['Family'].loc[test_df['Family'] > 0] = 1

test_df['Family'].loc[test_df['Family'] == 0] = 0



train_df['Family'].loc[(train_df['Age'] < 16) & (train_df['Family'] == 0)] = 2

test_df['Family'].loc[(test_df['Age'] < 16) & (test_df['Family'] == 0)] = 2



train_df['Survived'].groupby([train_df['Sex'], train_df['Family']]).mean().unstack()
#__________________Embarked______________________________#

train_df["Embarked"] = train_df["Embarked"].fillna("S")

#train_df['Survived'].groupby(train_df['Embarked']).mean()

train_df['Embarked'].loc[train_df['Embarked'] == 'S'] = 0

train_df['Embarked'].loc[train_df['Embarked'] == 'C'] = 1

train_df['Embarked'].loc[train_df['Embarked'] == 'Q'] = 2



test_df['Embarked'].loc[test_df['Embarked'] == 'S'] = 0

test_df['Embarked'].loc[test_df['Embarked'] == 'C'] = 1

test_df['Embarked'].loc[test_df['Embarked'] == 'Q'] = 2



#___________________________Fare_________________________#

test_df["Fare"].fillna(test_df["Fare"].median(), inplace=True)

train_df['Fare'] = train_df['Fare'].astype(int)

test_df['Fare']    = test_df['Fare'].astype(int)



#____________Drop columns not required____________________#



train_df = train_df.drop(['PassengerId','Name','Ticket','SibSp','Parch','Cabin'], axis=1)

test_df = test_df.drop(['Name','Ticket','Cabin','SibSp','Parch'], axis=1)

#train_df.info()

#test_df.info()

#________________define training and testing sets_______________#



X_train = train_df.drop("Survived",axis=1)

Y_train = train_df["Survived"]

X_test  = test_df.drop("PassengerId",axis=1).copy()







# Logistic Regression

logreg = LogisticRegression()

logreg.fit(X_train, Y_train)

Y_pred = logreg.predict(X_test)

LR_Score = logreg.score(X_train, Y_train)



# Random Forests

random_forest = RandomForestClassifier(n_estimators=100)

random_forest.fit(X_train, Y_train)

Y_pred = random_forest.predict(X_test)

RF_Score = random_forest.score(X_train, Y_train)



#___________Submission______________________________#



submission = pd.DataFrame({

        "PassengerId": test_df["PassengerId"],

        "Survived": Y_pred

    })

submission.to_csv('Prediction_Output.csv', index=False)



# Support Vectore Machine

svc = SVC()

svc.fit(X_train, Y_train)

Y_pred = svc.predict(X_test)

SV_Score = svc.score(X_train, Y_train)





# K Neighbors

knn = KNeighborsClassifier(n_neighbors = 3)

knn.fit(X_train, Y_train)

Y_pred = knn.predict(X_test)

KN_Score = knn.score(X_train, Y_train)



# Decision Tree

decision_tree = DecisionTreeClassifier()

decision_tree.fit(X_train, Y_train)

Y_pred = decision_tree.predict(X_test)

DT_Score = decision_tree.score(X_train, Y_train)





#_____________Model Comparison________________________________#

models = pd.DataFrame({

    'Model': ['Logistic Regression', 

              'Random Forest', 'Support Vector Machine','KNN',

              'Decision Tree'],

    'Score': [LR_Score, RF_Score, SV_Score, KN_Score, DT_Score]})

models.sort_values(by='Score', ascending=False)




