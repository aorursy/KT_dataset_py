import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import warnings

warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn import preprocessing

from sklearn.metrics import accuracy_score

import gc

import time

from contextlib import contextmanager
@contextmanager

def timer(title):

    t0 = time.time()

    yield

    print("{} - done in {:.0f}s".format(title, time.time() - t0))
def data_preprocessing():



    print("Data Preprocessing Process Has Been Started" "\n")



    train = pd.read_csv("../input/titanic/train.csv")

    test = pd.read_csv("../input/titanic/test.csv")



    train = train.drop(['Ticket'], axis = 1)

    test = test.drop(['Ticket'], axis = 1)



    train['Fare'] = train['Fare'].replace(512.3292, 300)

    test['Fare'] = test['Fare'].replace(512.3292, 300)



    train["Age"] = train["Age"].fillna(train["Age"].median())

    test["Age"] = test["Age"].fillna(test["Age"].median())



    # Fill NA with the most frequent value:

    train["Embarked"] = train["Embarked"].fillna("S")

    test["Embarked"] = test["Embarked"].fillna("S")



    test["Fare"] = test["Fare"].fillna(12)



    train["CabinBool"] = train["Cabin"].notnull().astype('int')

    test["CabinBool"] = test["Cabin"].notnull().astype('int')



    train = train.drop(['Cabin'], axis = 1)

    test = test.drop(['Cabin'], axis = 1)



    # Map each Embarked value to a numerical value:



    embarked_mapping = {"S": 1, "C": 2, "Q": 3}



    train['Embarked'] = train['Embarked'].map(embarked_mapping)

    test['Embarked'] = test['Embarked'].map(embarked_mapping)





    lbe = preprocessing.LabelEncoder()





    train["Sex"] = lbe.fit_transform(train["Sex"])

    test["Sex"] = lbe.fit_transform(test["Sex"])



    train["Title"] = train["Name"].str.extract(' ([A-Za-z]+)\.', expand=False)

    test["Title"] = test["Name"].str.extract(' ([A-Za-z]+)\.', expand=False)



    train['Title'] = train['Title'].replace(['Lady', 'Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Jonkheer', 'Dona'], 'Rare')

    train['Title'] = train['Title'].replace(['Countess', 'Lady', 'Sir'], 'Royal')

    train['Title'] = train['Title'].replace('Mlle', 'Miss')

    train['Title'] = train['Title'].replace('Ms', 'Miss')

    train['Title'] = train['Title'].replace('Mme', 'Mrs')





    test['Title'] = test['Title'].replace(['Lady', 'Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Jonkheer', 'Dona'], 'Rare')

    test['Title'] = test['Title'].replace(['Countess', 'Lady', 'Sir'], 'Royal')

    test['Title'] = test['Title'].replace('Mlle', 'Miss')

    test['Title'] = test['Title'].replace('Ms', 'Miss')

    test['Title'] = test['Title'].replace('Mme', 'Mrs')



    # Map each of the title groups to a numerical value



    title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Royal": 5, "Rare": 5}

    train['Title'] = train['Title'].map(title_mapping)

    test['Title'] = test['Title'].map(title_mapping)



    train = train.drop(['Name'], axis = 1)

    test = test.drop(['Name'], axis = 1)





    bins = [0, 5, 12, 18, 24, 35, 60, np.inf]



    mylabels = ['Baby', 'Child', 'Teenager', 'Student', 'Young Adult', 'Adult', 'Senior']





    train['AgeGroup'] = pd.cut(train["Age"], bins, labels = mylabels)

    test['AgeGroup'] = pd.cut(test["Age"], bins, labels = mylabels)





    # Map each Age value to a numerical value:

    age_mapping = {'Baby': 1, 'Child': 2, 'Teenager': 3, 'Student': 4, 'Young Adult': 5, 'Adult': 6, 'Senior': 7}

    train['AgeGroup'] = train['AgeGroup'].map(age_mapping)

    test['AgeGroup'] = test['AgeGroup'].map(age_mapping)



    #dropping the Age feature for now, might change:

    train = train.drop(['Age'], axis = 1)

    test = test.drop(['Age'], axis = 1)



    # Map Fare values into groups of numerical values:

    train['FareBand'] = pd.qcut(train['Fare'], 4, labels = [1, 2, 3, 4])

    test['FareBand'] = pd.qcut(test['Fare'], 4, labels = [1, 2, 3, 4])



    # Drop Fare values:

    train = train.drop(['Fare'], axis = 1)

    test = test.drop(['Fare'], axis = 1)



    print("Data Preprocessing Process Has Been Finished" "\n")

    

    return train, test
def feature_engineering(train, test):



    print("Feature Engineering Process Has Been Started" "\n")



    train["FamilySize"] = train["SibSp"] + train["Parch"] + 1

    test["FamilySize"] = train["SibSp"] + train["Parch"] + 1



    # Create new feature of family size:



    train['Single'] = train['FamilySize'].map(lambda s: 1 if s == 1 else 0)

    train['SmallFam'] = train['FamilySize'].map(lambda s: 1 if  s == 2  else 0)

    train['MedFam'] = train['FamilySize'].map(lambda s: 1 if 3 <= s <= 4 else 0)

    train['LargeFam'] = train['FamilySize'].map(lambda s: 1 if s >= 5 else 0)



    # Create new feature of family size:



    test['Single'] = test['FamilySize'].map(lambda s: 1 if s == 1 else 0)

    test['SmallFam'] = test['FamilySize'].map(lambda s: 1 if  s == 2  else 0)

    test['MedFam'] = test['FamilySize'].map(lambda s: 1 if 3 <= s <= 4 else 0)

    test['LargeFam'] = test['FamilySize'].map(lambda s: 1 if s >= 5 else 0)



    # Convert Title and Embarked into dummy variables:



    train = pd.get_dummies(train, columns = ["Title"], drop_first = True)

    train = pd.get_dummies(train, columns = ["Embarked"], drop_first = True, prefix="Em")



    test = pd.get_dummies(test, columns = ["Title"], drop_first = True)

    test = pd.get_dummies(test, columns = ["Embarked"], drop_first = True, prefix="Em")



    # Create categorical values for Pclass:

    train["Pclass"] = train["Pclass"].astype("category")

    train = pd.get_dummies(train, columns = ["Pclass"],prefix="Pc")



    test["Pclass"] = test["Pclass"].astype("category")

    test = pd.get_dummies(test, columns = ["Pclass"],prefix="Pc")



    print("Feature Engineering Process Has Been Finished" "\n")

    

    

    return train, test
def modeling(train):



    print("Modeling Process Has Been Started:" "\n")



    X = train.drop(['Survived', 'PassengerId'], axis=1)

    Y = train["Survived"]



    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.20, random_state = 17)



    from sklearn.ensemble import GradientBoostingClassifier



    gbm = GradientBoostingClassifier()



    gbm_params = {

            'n_estimators': [200, 500],

            'subsample': [1.0],

            'max_depth': [8],

            'learning_rate': [0.01,0.02],

            "min_samples_split": [10]}



    gbm_cv_model = GridSearchCV(gbm, gbm_params, cv = 10, n_jobs = -1, verbose = 5)



    gbm_cv_model.fit(x_train, y_train)



    print(gbm_cv_model.best_params_ , "\n")



    gbm_tuned = GradientBoostingClassifier(learning_rate = gbm_cv_model.best_params_["learning_rate"], 

                        max_depth = gbm_cv_model.best_params_["max_depth"],

                        min_samples_split = gbm_cv_model.best_params_["min_samples_split"],

                        n_estimators = gbm_cv_model.best_params_["n_estimators"],

                        subsample = gbm_cv_model.best_params_["subsample"])



    gbm_tuned.fit(x_train, y_train)



    y_pred = gbm_tuned.predict(x_test)

    print("Accuracy Score of Your Model:")

    print(round(accuracy_score(y_pred, y_test) * 100, 2))

    

    return gbm_tuned

def submission(gbm_tuned, test):



    #set ids as PassengerId and predict survival 

    ids = test['PassengerId']



    predictions = gbm_tuned.predict(test.drop('PassengerId', axis=1))



    #set the output as a dataframe and convert to csv file named submission.csv

    output = pd.DataFrame({ 'PassengerId' : ids, 'Survived': predictions })

    

    output.to_csv('submission.csv', index=False)

    print("Submission file has been created")
def main():

    

    with timer("Pre processing Time"):

        train, test = data_preprocessing()

    

    with timer("Feature Engineering"):

        train, test = feature_engineering(train, test)

        

    with timer("Modeling"):

        gbm_tuned = modeling(train)

        

    with timer("Submission"):

        submission(gbm_tuned, test)    
if __name__ == "__main__":

    with timer("Full model run"):

        main()