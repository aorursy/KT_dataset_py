#Titanic Challange . How to start?

#I got 0.75 score
import numpy as np

import pandas as pd

import os

import matplotlib.pyplot



#os.getcwd()

#get your path

#os.chdir('/Users/Administrator/Desktop/MachineLearningOnKaggle/Titanic Challenger')

#os.getcwd()
#input data

train_df = pd.read_csv('../input/train.csv')

test_df = pd.read_csv('../input/test.csv')

train_df.head()

test_df.head()
train_df
#Preprocessing  data  , treating null value

# on test_df dataset

sex_mode_testdf = test_df["Sex"].mode()

pclass_median_testdf = test_df["Pclass"].median()

sibsp_mode_testdf = test_df["SibSp"].median()

fare_mean_testdf = test_df["Fare"].mean()

embarked_mode_testdf = test_df["Embarked"].mode()

age_mean_testdf = test_df["Age"].mean()



test_df["Sex"] = test_df["Sex"].replace({None : sex_mode_testdf})

test_df["Pclass"] = test_df["Pclass"].replace({None : pclass_median_testdf})

test_df["SibSp"] = test_df["SibSp"].replace({None : sibsp_mode_testdf})

test_df["Fare"] = test_df["Fare"].replace({None : fare_mean_testdf})

test_df["Embarked"] = test_df["Embarked"].replace({None : embarked_mode_testdf})

test_df["Age"] = test_df["Age"].replace({None : age_mean_testdf})



test_df["Sex"][test_df["Sex"] == 'male'] = 0

test_df["Sex"][test_df["Sex"] == 'female'] = 1



# on train_df dataset

sex_mode = train_df["Sex"].mode()

pclass_median = train_df["Pclass"].median()

sibsp_mode = train_df["SibSp"].median()

fare_mean = train_df["Fare"].mean()

embarked_mode = train_df["Embarked"].mode()

age_mean = train_df["Age"].mean()

parch_median = train_df["Parch"].median()



train_df["Sex"] = train_df["Sex"].replace({None : sex_mode})

train_df["Pclass"] = train_df["Pclass"].replace({None : pclass_median})

train_df["SibSp"] = train_df["SibSp"].replace({None : sibsp_mode})

train_df["Fare"] = train_df["Fare"].replace({None : fare_mean})

train_df["Embarked"] = train_df["Embarked"].replace({None : embarked_mode})

train_df["Age"] = train_df["Age"].replace({None : age_mean})

train_df["Parch"] = train_df["Parch"].replace({None : parch_median})

train_df["Sex"][train_df["Sex"] == 'male'] = 0

train_df["Sex"][train_df["Sex"] == 'female'] = 1
#visualize data

train_df["Survived"].value_counts()

#tranform categorical variables

train_df["Sex"][train_df["Sex"] == 'male'] = 0

train_df["Sex"][train_df["Sex"] == 'female'] = 1



train_df["Embarked"] = train_df["Embarked"].fillna(train_df['Enbarked'].mode())



train_df["Embarked"][train_df["Embarked"] == "S"] = 0

train_df["Embarked"][train_df["Embarked"] == "C"] = 1

train_df["Embarked"][train_df["Embarked"] == "Q"] = 2







# Create the target and features numpy arrays: target, features_one

target = train_df["Survived"].values

#features_one = train_df[["Pclass","Sex","Age"]].values

features_one = train_df[["Sex","Age","SibSp","Pclass","Fare"]].values

#create model and training model

#select decission tree model for first training

from sklearn import tree



my_tree_one = tree.DecisionTreeClassifier()

my_tree_one.fit(features_one, target)



print(my_tree_one.score(features_one, target))

#create test feature for predict

test_features = test_df[["Sex","Age","SibSp","Pclass","Fare"]].values

#predict

my_prediction = my_tree_one.predict(test_features)

#create submition and save file

submission = pd.DataFrame({

    "PassengerId": test_df["PassengerId"],

    "Survived": my_prediction

})

submission.to_csv("submission8.csv", index=False)
#second prediction with random forest

#create feature and target



train_df_two = train_df.copy()

train_df_two["family_size"] = train_df_two["SibSp"] + train_df_two["Parch"] + 1

features_two =  train_df_two[["Sex","Age","SibSp","Pclass","Parch","Fare","family_size"]].values



test_df_two  = test_df.copy()

test_df_two["family_size"] = test_df_two["SibSp"] + test_df_two["Parch"] + 1

test_features_two =  test_df_two[["Sex","Age","SibSp","Pclass","Parch","Fare","family_size"]].values





#create model and predict

from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(max_depth = 10, min_samples_split=2, n_estimators = 100, random_state = 1) 

clf = clf.fit(features_one ,target)

clf_prediction = clf.predict(test_features)

print(clf.score(features_one , target))
#submittion 2

submission = pd.DataFrame({

    "PassengerId": test_df["PassengerId"],

    "Survived": clf_prediction

})

submission.to_csv("submission8_randomforest.csv", index=False)