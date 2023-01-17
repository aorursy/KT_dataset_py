# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from matplotlib import pyplot as plt



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df_train = pd.read_csv("/kaggle/input/titanic/train.csv")

df_train.head()
df_train.info()
df_survived = df_train.groupby("Survived").agg({'PassengerId': 'count'})

df_pclass = df_train.groupby("Pclass").agg({'PassengerId': 'count'})

df_sex = df_train.groupby("Sex").agg({'PassengerId': 'count'})

df_embarked = df_train.groupby("Embarked").agg({'PassengerId': 'count'})





fig, ax = plt.subplots(2,2,figsize=(8,8))



ax[0][0].bar([0,1],df_survived["PassengerId"],tick_label = ["no","yes"])

ax[0][0].set_title("Survived?")



ax[0][1].bar([0,1,2],df_pclass["PassengerId"],tick_label = ["1st","2nd","3rd"])

ax[0][1].set_title("Passenger class")



ax[1][0].bar([0,1],df_sex["PassengerId"],tick_label = ["female","male"])

ax[1][0].set_title("Sex")



ax[1][1].bar([0,1,2],df_embarked["PassengerId"],tick_label = ["Cherbourg","Queenstown","Southampton"])

ax[1][1].set_title("Port of Embarkation")





plt.show()
fig, ax = plt.subplots(2,2,figsize=(8,8))



df_train["Age"].hist(ax=ax[0][0])

ax[0][0].set_xlabel("Age")



df_train["SibSp"].hist(ax=ax[0][1])

ax[0][1].set_xlabel("Number of siblings aboard")



df_train["Parch"].hist(ax=ax[1][0])

ax[1][0].set_xlabel("Number of parents or children aboard")



df_train["Fare"].hist(ax=ax[1][1])

ax[1][1].set_xlabel("Fare")



plt.show()
df_train["Female"] = df_train["Sex"] == "female"
from sklearn import preprocessing

import numpy as np
train_standard = preprocessing.scale(df_train[["Pclass","Age","SibSp","Parch","Fare","Female"]])



#we set the missing ages to the mean (0)

train_standard = np.nan_to_num(np.append(train_standard, np.array([df_train["Survived"]]).T, axis = 1))



#dataframe for pretty printing

train_standard_df = pd.DataFrame(train_standard)

train_standard_df.columns = ["Pclass","Age","SibSp","Parch","Fare","Female","Survived"]
train_standard_df.corr()
df_train[df_train["Pclass"]==1]["Fare"].hist()

plt.show()
train_standard_child = preprocessing.scale(df_train[df_train["Age"]<12][["Pclass","Age","SibSp","Parch","Fare","Female"]])



#we set the missing ages to the mean (0)

train_standard_child = np.nan_to_num(np.append(train_standard_child, np.array([df_train[df_train["Age"]<12]["Survived"]]).T, axis = 1))



#dataframe for pretty printing

train_standard_child_df = pd.DataFrame(train_standard_child)

train_standard_child_df.columns = ["Pclass","Age","SibSp","Parch","Fare","Female","Survived"]
train_standard_child_df.corr()
train_standard_adult = preprocessing.scale(df_train[(df_train["Age"]>18)&(df_train["Parch"]>0)][["Pclass","Age","SibSp","Parch","Fare","Female"]])



#we set the missing ages to the mean (0)

train_standard_adult = np.nan_to_num(np.append(train_standard_adult, np.array([df_train[(df_train["Age"]>18)&(df_train["Parch"]>0)]["Survived"]]).T, axis = 1))



#dataframe for pretty printing

train_standard_adult_df = pd.DataFrame(train_standard_adult)

train_standard_adult_df.columns = ["Pclass","Age","SibSp","Parch","Fare","Female","Survived"]
train_standard_adult_df.corr()
df_train["is_parent"] = (df_train["Age"]>18)&(df_train["Parch"]>0)

df_train["is_child"]  = df_train["Age"]<12
train_standard = preprocessing.scale(df_train[["Pclass","Age","SibSp","Parch","Fare","Female","is_parent","is_child"]])



#we set the missing ages to the mean (0)

train_standard = np.nan_to_num(np.append(train_standard, np.array([df_train["Survived"]]).T, axis = 1))



#dataframe for pretty printing

train_standard_df = pd.DataFrame(train_standard)

train_standard_df.columns = ["Pclass","Age","SibSp","Parch","Fare","Female","is_parent","is_child","Survived"]
train_standard_df.corr()
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import cross_val_score



classifier = LogisticRegression()



scores = cross_val_score(classifier, train_standard[:,:-1], train_standard[:,-1], cv=10)



scores.mean()
classifier = LogisticRegression().fit(train_standard[:,:-1], train_standard[:,-1])

classifier.coef_
df_test = pd.read_csv("/kaggle/input/titanic/test.csv")

df_test["is_parent"] = (df_test["Age"]>18)&(df_test["Parch"]>0)

df_test["is_child"]  = df_test["Age"]<12

df_test["Female"] = df_test["Sex"] == "female"

test_standard = np.nan_to_num(preprocessing.scale(df_test[["Pclass","Age","SibSp","Parch","Fare","Female","is_parent","is_child"]]))



predictions = classifier.predict(test_standard).astype(int)

df_test["Survived"] = predictions



df_test[["PassengerId","Survived"]].to_csv("submission.csv",index=False)
from sklearn.svm import SVC

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.tree import DecisionTreeClassifier

SVM_classifier = SVC(gamma="auto")

KNN_classifier = KNeighborsClassifier(n_neighbors=3)

RF_classifier = RandomForestClassifier(max_depth=5)

DT_classifier = DecisionTreeClassifier()

score = {}

score["Log Reg"] = scores.mean()

score["SVM"] = cross_val_score(SVM_classifier, train_standard[:,:-1], train_standard[:,-1], cv=10).mean()

score["KNN"] = cross_val_score(KNN_classifier, train_standard[:,:-1], train_standard[:,-1], cv=10).mean()

score["Random Forest"] = cross_val_score(RF_classifier, train_standard[:,:-1], train_standard[:,-1], cv=10).mean()

score["Decision Tree"] = cross_val_score(DT_classifier, train_standard[:,:-1], train_standard[:,-1], cv=10).mean()

score
n_neighbours = [1,2,3,5,8,10,15,20]

max_depth = [1,2,3,5,8,10,15,20]

tuning_scores_KNN = {}

tuning_scores_RF = {}



for nn in n_neighbours:

    KNN_classifier = KNeighborsClassifier(n_neighbors=nn)

    tuning_scores_KNN[nn] = cross_val_score(KNN_classifier, train_standard[:,:-1], train_standard[:,-1], cv=10).mean()



for md in max_depth:

    RF_classifier = RandomForestClassifier(max_depth=md)

    tuning_scores_RF[md] = cross_val_score(RF_classifier, train_standard[:,:-1], train_standard[:,-1], cv=10).mean()

tuning_scores_KNN
tuning_scores_RF
df_test = pd.read_csv("/kaggle/input/titanic/test.csv")

df_test["is_parent"] = (df_test["Age"]>18)&(df_test["Parch"]>0)

df_test["is_child"]  = df_test["Age"]<12

df_test["Female"] = df_test["Sex"] == "female"

test_standard = np.nan_to_num(preprocessing.scale(df_test[["Pclass","Age","SibSp","Parch","Fare","Female","is_parent","is_child"]]))



classifier = RandomForestClassifier(max_depth=5)

classifier.fit(train_standard[:,:-1], train_standard[:,-1])

predictions = classifier.predict(test_standard).astype(int)

df_test["Survived"] = predictions



df_test[["PassengerId","Survived"]].to_csv("submission_RF.csv",index=False)