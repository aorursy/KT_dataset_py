# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from matplotlib import pyplot as plt #data plotting

import seaborn as sns #data visulization

from sklearn.svm import SVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.naive_bayes import GaussianNB

from sklearn.neighbors import KNeighborsClassifier







# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
#load test and training dataset



titanic_df = pd.read_csv("../input/train.csv")

test_df    = pd.read_csv("../input/test.csv")

print (titanic_df.head())

print(titanic_df.info())

print("----------------")

print(test_df.info())

#by printing the info we identify the null objects present in the dataset
#Now, the most important part: Looking at the dataset and dropping the features, 

#name is not important

#for training set ID is not important but for test we need to keep it because submission requires it

#ticket number is not important

#So, we drop these features..

titanic_df = titanic_df.drop(['PassengerId','Name','Ticket'],axis=1)

test_df=test_df.drop(['Name','Ticket'], axis=1)

titanic_df = titanic_df[pd.notnull(titanic_df['Embarked'])]





titanic_df.columns

#now age has some null values, which need to be filled, so, we try to fill them with random int values

#between mean(age)-std(age) and mean(age)+std(age)

random_1=np.random.randint(titanic_df["Age"].mean()-titanic_df["Age"].std(), titanic_df["Age"].mean()+titanic_df["Age"].std(), size= titanic_df["Age"].isnull().sum())



random_2=np.random.randint(test_df["Age"].mean()-test_df["Age"].std(), test_df["Age"].mean()+test_df["Age"].std(), size= test_df["Age"].isnull().sum())

titanic_df["Age"][np.isnan(titanic_df["Age"])] = random_1

test_df["Age"][np.isnan(test_df["Age"])] = random_2

#cabin has a lot of Nan value. Though it is important it should be dropped

titanic_df=titanic_df.drop(["Cabin"], axis=1)

test_df=test_df.drop(["Cabin"], axis=1)
#checking the info now

titanic_df.info()
#Fare, we need to fill 2 nan values in fare column of test_df, we do that by filling median of data in these places

test_df["Fare"].fillna(test_df["Fare"].median(), inplace=True)

titanic_df["Fare"]=titanic_df["Fare"].astype(int)

test_df["Fare"]=test_df["Fare"].astype(int)
#now we need to convert sex and embarked in classification value(0,1,2,...)

titanic_df["Sex"].loc[titanic_df["Sex"]=="male"]=1

titanic_df["Sex"].loc[titanic_df["Sex"]=="female"]=0



test_df["Sex"].loc[test_df["Sex"]=="male"]=1

test_df["Sex"].loc[test_df["Sex"]=="female"]=0
#converting embarked values

titanic_df["Embarked"].unique()

titanic_df["Embarked"].loc[titanic_df["Embarked"]=="C"]=0

titanic_df["Embarked"].loc[titanic_df["Embarked"]=="S"]=1

titanic_df["Embarked"].loc[titanic_df["Embarked"]=="Q"]=2





test_df["Embarked"].loc[test_df["Embarked"]=="C"]=0

test_df["Embarked"].loc[test_df["Embarked"]=="S"]=1

test_df["Embarked"].loc[test_df["Embarked"]=="Q"]=2

#preparing X_train, Y_train and X_test

Y_train=titanic_df["Survived"]

X_train=titanic_df.drop(["Survived"], axis=1)



X_test=test_df.drop(["PassengerId"], axis=1)



Y_train.shape
#preparing SVM classifier

#SVM_classifier=SVC()

#SVM_classifier.fit(X_train, Y_train)

#Y_pred = SVM_classifier.predict(X_test)

#SVM_classifier.score(X_train, Y_train)
# preparing Logistic Regression



#logreg = LogisticRegression()

#logreg.fit(X_train, Y_train)

#Y_pred = logreg.predict(X_test)

#logreg.score(X_train, Y_train)
# preparing Random Forests classifier



random_forest = RandomForestClassifier(n_estimators=100)

random_forest.fit(X_train, Y_train)

Y_pred = random_forest.predict(X_test)

random_forest.score(X_train, Y_train)
##knn = KNeighborsClassifier(n_neighbors = 3)

#knn.fit(X_train, Y_train)

#Y_pred = knn.predict(X_test)

#knn.score(X_train, Y_train)
#preparing Gaussian Naive Bayes

#gaussian = GaussianNB()

#gaussian.fit(X_train, Y_train)

#Y_pred = gaussian.predict(X_test)

#gaussian.score(X_train, Y_train)
#we see that Random Forests has the best score and so we choose random forests as our main classifier finally

submission = pd.DataFrame({

        "PassengerId": test_df["PassengerId"],

        "Survived": Y_pred

    })

submission.to_csv('titanic.csv', index=False)