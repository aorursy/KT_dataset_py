# coding: utf-8

#to do

#1: A feature amount is generated from each item. Ticket and Cabin are excluded this time.

#2: Use the grid search to get the best parameters.

#3: Make data for submission using the best result parameter.



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



train = pd.read_csv("../input/train.csv")

test_data = pd.read_csv("../input/test.csv")

test_result = pd.read_csv("../input/genderclassmodel.csv")



#For data processing, make train, test and answer all together and make over 1300 data sets. (To divide later)



test = pd.merge(test_data, test_result, how="outer", on="PassengerId")

df = pd.concat([train, test], axis=0).reset_index(drop=True)

df.head()
#1: A feature amount is generated from each item. Ticket and Cabin are excluded this time.



df = pd.get_dummies(df, columns=["Embarked"], prefix="Em")

df = pd.get_dummies(df, columns=["Sex"])

df.drop(labels=["Ticket","Cabin"], axis=1, inplace=True)

df.head()
#Detects the title from the name and converts it into the feature quantity

df_title = [i.split(",")[1].split(".")[0].strip() for i in df["Name"]]

df["Title"] = pd.Series(df_title)

#df["Title"].value_counts() #Because Mlle looked like Mile and made a spelling mistake, for confirmation

g = sns.countplot(x="Title", data=df)

g = plt.setp(g.get_xticklabels(), rotation=45)
df["Title"] = df["Title"].replace(['the Countess','Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

df["Title"] = df["Title"].map({"Master":0, "Miss":1, "Ms":1, "Mme":1, "Mlle":1, "Lady":1, "Mrs":2, "Mr":3, "Rare":4})

df["Title"] = df["Title"].astype('int')

df.drop(labels=["Name"], axis=1, inplace=True)



df.head()
#Fill Fare missing values with the median value

df["Fare"] = df["Fare"].fillna(df["Fare"].median())



#Apply log to Fare to reduce skewness distribution

df["Fare"] = df["Fare"].map(lambda i: np.log(i) if i > 0 else 0)



df.head()
#Missing value of age is filled with average

df["Age"] = df["Age"].fillna(df["Age"].mean())
#Repartition the data set back to the training set and test set.



X_train = pd.DataFrame(df[:len(train)])

X_test = pd.DataFrame(df[len(train):])



Y_train = pd.Series(X_train["Survived"])

X_train.drop(labels=["Survived","PassengerId"], axis=1, inplace=True)



X_test.drop(labels=["Survived","PassengerId"], axis=1, inplace=True)



X_train.head()
#2: Use the grid search to get the best parameters.



from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, learning_curve
RFC = RandomForestClassifier()

RF_Param_Grid = {

    "max_depth": [4,8,16,32],

    "min_samples_split": [2,4,8,16],

    "min_samples_leaf": [1,3],

    "bootstrap": [False],

    "n_estimators": [50,100],

    "criterion": ["gini"]

}

kfold = StratifiedKFold(n_splits=10)



g_clf = GridSearchCV(RFC, param_grid=RF_Param_Grid, cv=kfold, scoring="accuracy", n_jobs=4, verbose=1)

g_clf.fit(X_train, Y_train)
g_clf.best_score_
g_clf.best_params_
#3: Make data for submission using the best result parameter.



bp = g_clf.best_params_



clf = RandomForestClassifier(

    bootstrap=bp["bootstrap"], 

    criterion=bp["criterion"], 

    max_depth=bp["max_depth"], 

    min_samples_leaf=bp["min_samples_leaf"], 

    min_samples_split=bp["min_samples_split"], 

    n_estimators=bp["n_estimators"]

)

clf.fit(X_train, Y_train)
IDtest = test_data["PassengerId"]

test_Survived = pd.Series(clf.predict(X_test), name="Survived")

results = pd.concat([IDtest, test_Survived], axis=1)
# score: 0.80382