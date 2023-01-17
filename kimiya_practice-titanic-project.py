# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
# read data sheets

titanic = pd.read_csv("../input/train.csv")



#Fill missing value in the column Age with median

titanic["Age"]=titanic["Age"].fillna(titanic["Age"].median())

#print(titanic["Age"])

#print(titanic["Sex"].unique())



#Replace Sex with digit

titanic.loc[titanic["Sex"] == "male", "Sex"] = 0

titanic.loc[titanic["Sex"] == "female", "Sex"] = 1



#Replace Embarked with digit

#print(titanic["Embarked"].unique())

titanic["Embarked"] = titanic["Embarked"].fillna("S")

titanic.loc[titanic["Embarked"] == "S", "Embarked"] = 0

titanic.loc[titanic["Embarked"] == "C", "Embarked"] = 1

titanic.loc[titanic["Embarked"] == "Q", "Embarked"] = 2

#print(titanic["Embarked"])
#Finished preparing data

#Start machine learning now

from sklearn.linear_model import LinearRegression

from sklearn.cross_validation import KFold



#Columsn to be used for predicting the target

predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]



#Initializing algorithm

alg = LinearRegression()



#Generate cross validation folds

kf = KFold(titanic.shape[0], n_folds = 3, random_state = 1)

#print(kf)

predictions = []

for train, test in kf:

    train_predictors = (titanic[predictors].iloc[train, :])

    #print(train_predictors)

    train_target = titanic["Survived"].iloc[train]

    #print(train_target)

    alg.fit(train_predictors, train_target)

    #print(alg.fit(train_predictors, train_target))

    test_predictions = alg.predict(titanic[predictors].iloc[test,:])

    #print(test_predictions)

    predictions.append(test_predictions)

    #print(predictions)

    

#Combine the predictions which are numpy arrays

predictions = np.concatenate(predictions, axis=0)

predictions[predictions > 0.5] = 1

predictions[predictions <= 0.5] = 0

accuracy = sum(predictions[predictions == titanic["Survived"]])/len(titanic["Survived"])



#Survived = np.array(titanic["Survived"])

#rightly_predicted = np.where(predictions == Survived)

#accuracy = len(rightly_predicted[0])/len(titanic["Survived"])

print(accuracy)
#Improve the prediction with logistic regression

from sklearn import cross_validation

from sklearn.linear_model import LogisticRegression

 

#Initialize the algorithm

alg = LogisticRegression(random_state=1)



#Computing the accuracy score for all the cross validation folds.

scores = cross_validation.cross_val_score(alg, titanic[predictors], titanic["Survived"], cv = 3)



print(scores.mean())
#Procee titanic_test in the same way with titanic processed.

titanic_test = pd.read_csv('../input/test.csv')

titanic_test["Age"] =  titanic_test["Age"].fillna(titanic["Age"].median())

titanic_test.loc[titanic_test["Sex"] == "male", "Sex"] = 0

titanic_test.loc[titanic_test["Sex"] == "female", "Sex"] = 1

titanic_test["Embarked"] = titanic_test["Embarked"].fillna("S")

titanic_test.loc[titanic_test["Embarked"] == "S", "Embarked"] = 0

titanic_test.loc[titanic_test["Embarked"] == "C", "Embarked"] = 1

titanic_test.loc[titanic_test["Embarked"] == "Q", "Embarked"] = 2



titanic_test["Fare"] = titanic_test["Fare"].fillna(titanic_test["Fare"].median())
#Submit the prediction result



#Initialize the algorithm class

alg = LogisticRegression(random_state = 1)



#Train the algorithm using all the training data

#print(predictors)

alg.fit(titanic[predictors], titanic["Survived"])



#Make predictions with the test set

predictions = alg.predict(titanic_test[predictors])

#print(predictions)



#Create a new dataframe with only the columns Kaggle wants from the dataset

submission = pd.DataFrame({

    "PassengerId": titanic_test["PassengerId"],

    "Survived": predictions

})

#print(submission)

#submission.to_csv("first_titanic_practice.csv", index = False)
#Improving the model with random forest 

from sklearn import cross_validation

from sklearn.ensemble import RandomForestClassifier



predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]

alg = RandomForestClassifier(random_state = 1,

                            n_estimators = 10, #num of trees 

                            min_samples_split = 2, #min num of row to split

                            min_samples_leaf = 1 #min of samples at a branch

                            )



#Generate cross validation folds

kf_rf = cross_validation.KFold(titanic.shape[0], n_folds = 3, random_state = 1)

#Generate cross validation prediction

scores = cross_validation.cross_val_score(alg, titanic[predictors],titanic["Survived"], cv = kf_rf )

#print(scores)

scores = scores.mean()

print(scores)
#Tweak the parameter of the random forest

alg = RandomForestClassifier(random_state = 1, n_estimators = 50, min_samples_split = 4, min_samples_leaf = 2)



kf_rf_tweaked = cross_validation.KFold(titanic.shape[0], n_folds = 3, random_state = 1)

scores = cross_validation.cross_val_score(alg, titanic[predictors], titanic["Survived"], cv = kf_rf_tweaked)



scores = scores.mean()

print(scores)
#Add new features

titanic["FamilySize"] = titanic["SibSp"] + titanic["Parch"]

titanic["NameLength"] = titanic["Name"].apply(lambda x: len(x))
#Data cleansing with regex

import re



def get_title(name):

    title_search = re.search(' ([A-Za-z]+)\.', name)

    if title_search:

        return title_search.group(1)

    return ""



#Get all the titles and find out how often each one occurs

titles = titanic["Name"].apply(get_title)

#print(pd.value_counts(titles))



#Map each title to an integer

title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Dr": 5, "Rev": 6, "Major": 7, "Col": 7, "Mlle": 8, "Mme": 8, "Don": 9, "Lady": 10, "Countess": 10, "Jonkheer": 10, "Sir": 9, "Capt": 7, "Ms": 2}

for key, value in title_mapping.items():

    titles[titles == key] = value

#print(titles)

titanic["Title"] = titles   
#Define family size

import operator



family_id_mapping = {}

def get_family_id(row):

    last_name = row["Name"].split(",")[0]

    

    family_id = "{0}{1}".format(last_name, row["FamilySize"])

    

    if family_id not in family_id_mapping:

        if len(family_id_mapping) == 0:

            current_id = 1

        else:

            current_id = (max(family_id_mapping.items(), key = operator.itemgetter(1))[1] + 1)

        family_id_mapping[family_id] = current_id

    

    return family_id_mapping[family_id]

 

family_ids = titanic.apply(get_family_id, axis = 1) 

family_ids[titanic["FamilySize"] < 3] = 1

#print(pd.value_counts(family_ids))



titanic["FamilyId"] = family_ids
#Finding The Best Features

#Check columns if correlate most closely with what to predict 

from sklearn.feature_selection import SelectKBest, f_classif

import matplotlib.pyplot as plt 

predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked", "FamilySize", "Title", "FamilyId", "NameLength"]



#Perform feature selection

#http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectKBest.html

selector = SelectKBest(f_classif, k = 5)

    #Where k = Number of top features to select

    #f_classif = ANOVA F-value between label/feature for classification tasks. 

    #It returns a pair of arrays (scores_, pvalues_) or a single array with scores.

selector.fit(titanic[predictors], titanic["Survived"])

#print(selector.pvalues_)

#print(selector.scores_)

scores = -np.log10(selector.pvalues_)

    #Here, the smaller p_value means the more the feature affect the target "Survived"



#Plot the scores of each feature

plt.bar(range(len(predictors)), scores)

plt.xticks(range(len(predictors)), predictors, rotation = "vertical")

plt.show()



#Pick the 4 best features

predictors = ["Pclass", "Sex", "Fare", "Title"]

alg = RandomForestClassifier(random_state = 1, n_estimators = 50,  min_samples_split = 8, min_samples_leaf = 4)

kf = cross_validation.KFold(titanic.shape[0], n_folds = 3, random_state = 1)

cv_scores = cross_validation.cross_val_score(alg, titanic[predictors], titanic["Survived"], cv = kf)



print(cv_scores.mean())
##One thing we can do to improve the accuracy of our predictions 

##is to ensemble different classifiers.

##generate predictions using information from a set of classifiers, instead of just one. 

##In practice, this means that we average their predictions.



from sklearn.ensemble import GradientBoostingClassifier

import numpy as np



algorithms = [

    [GradientBoostingClassifier(random_state = 1, n_estimators=25, max_depth = 3), ["Pclass", "Sex", "Age", "Fare", "Embarked", "FamilySize", "Title", "FamilyId"]],

    [LogisticRegression(random_state=1), ["Pclass", "Sex", "Fare", "FamilySize", "Title", "Age", "Embarked"]]

]  



#Initialize the cross validation folds

kf = KFold(titanic.shape[0], n_folds=3, random_state=1)



predictions = []

for train, test in kf:

    train_target = titanic["Survived"].iloc[train]

    full_test_predictions = []

#Make predictions for each algorithm on each fold

    for alg, predictors in algorithms:

        alg.fit(titanic[predictors].iloc[train, :], train_target)

    #Make prediction on the test fold

        test_predictions = alg.predict_proba(titanic[predictors].iloc[test,:].astype(float))[:,1]

        full_test_predictions.append(test_predictions)

        

#Emsembling by averaging the predictions

    test_predictions = (full_test_predictions[0] + full_test_predictions[1])/2

    test_predictions[test_predictions <= 0.5] = 0

    test_predictions[test_predictions > 0.5] = 1

    predictions.append(test_predictions)



#Concatenate all the predictions into one array

predictions = np.concatenate(predictions, axis = 0)

 

accuracy = sum(predictions[predictions == titanic["Survived"]])/len(predictions)

print(accuracy)



        

###Apply changes to test set

titles = titanic_test["Name"].apply(get_title)

title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Dr": 5, "Rev": 6, "Major": 7, "Col": 7, "Mlle": 8, "Mme": 8, "Don": 9, "Lady": 10, "Countess": 10, "Jonkheer": 10, "Sir": 9, "Capt": 7, "Ms": 2, "Dona": 10}



for key, value in title_mapping.items():

    titles[titles == key] = value

titanic_test["Title"] = titles

titanic_test["FamilySize"] = titanic_test["SibSp"] + titanic_test["Parch"]



family_ids = titanic_test.apply(get_family_id, axis = 1)

family_ids[titanic_test["FamilySize"] < 3] = -1

titanic_test["FamilyId"] = family_ids



titanic_test["NameLength"] = titanic_test["Name"].apply(lambda x: len(x))



###Predicting on the test set

predictors = ["Pclass", "Sex", "Age", "Fare", "Embarked", "FamilySize", "Title", "FamilyId"]



algorithms = [

    [GradientBoostingClassifier(random_state=1, n_estimators=25, max_depth=3), predictors],

    [LogisticRegression(random_state=1), ["Pclass", "Sex", "Fare", "FamilySize", "Title", "Age", "Embarked"]]

]



full_predictions = []

for alg, predictors in algorithms:

    alg.fit(titanic[predictors], titanic["Survived"])

    predictions = alg.predict_proba(titanic_test[predictors].astype(float))[:,1]

    full_predictions.append(predictions)

predictions = (full_predictions[0] * 3 + full_predictions[1])/4



predictions[predictions <= 0.5] = 0

predictions[predictions > 0.5] = 1

predictions = predictions.astype(int)

    

submission = pd.DataFrame({

    "passengerId": titanic_test["PassengerId"],

    "Survived": predictions

})



#print(submission)    

submission.to_csv("kaggle_practice.csv", index = False)
