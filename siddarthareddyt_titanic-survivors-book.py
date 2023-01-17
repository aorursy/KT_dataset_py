# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import sklearn.feature_selection

from sklearn.feature_selection import RFECV

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import cross_val_score

from sklearn.preprocessing import minmax_scale

from sklearn.svm import SVC

from sklearn.naive_bayes import GaussianNB

from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import StratifiedShuffleSplit

from sklearn.model_selection import ShuffleSplit

import keras

from keras.models import Sequential

from keras.layers import Dense

from sklearn import tree



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls","."]).decode("utf8"))



# Any results you write to the current directory are saved as output.
train = pd.read_csv("../input/train.csv")

holdout = pd.read_csv("../input/test.csv")



#no. of null embed columns

train.isnull().sum()



#some stats of the datasets

train.describe()

holdout.describe()

#data exploration



#survival vs gender indicates females have more chances of surviving

gender_pivot = train.pivot_table(index="Sex",values="Survived")

gender_pivot.plot.bar()

plt.show()





#pclass vs survived indicates class 1 and class 2 people are more likely to survive

class_pivot = train.pivot_table(index="Pclass", values="Survived")

class_pivot.plot.bar()

plt.show()



#Family size vs survived

family_cols = ["SibSp","Parch","Survived"]

family = train[family_cols].copy()

family['familysize'] = family[["SibSp","Parch"]].sum(axis=1)

familySize = family[["SibSp","Parch"]].sum(axis=1)

family["isalone"] = np.where(familySize>=1, 1, 0)

family_pivot = family.pivot_table(index="familysize",values="Survived")

isalone_pivot = family.pivot_table(index="isalone", values="Survived")

isalone_pivot.plot.bar(ylim=(0,1),yticks=np.arange(0,1,.1))

family_pivot.plot.bar(ylim=(0,1),yticks=np.arange(0,1,.1))

plt.show()
#data preprocessing

train["Fare"] = train["Fare"].fillna(train["Fare"].mean())

train["Embarked"] = train["Embarked"].fillna("S")



holdout["Fare"] = holdout["Fare"].fillna(train["Fare"].mean())

holdout["Embarked"] = holdout["Embarked"].fillna("S")



train["Age"] = train["Age"].fillna(-0.5)

holdout["Age"] = holdout["Age"].fillna(-0.5)



train.head(2)

holdout.head(2)



#feature engineering



#categorize age

cuts = [-1,0,5,12,18,35,60,100]

labels = ["Missing","Infant","Child","Teenager","Young Adult","Adult","Senior"]

train["Age_categories"] = pd.cut(train["Age"],cuts,labels=labels)

holdout["Age_categories"] = pd.cut(holdout["Age"],cuts,labels=labels)



#categorize fare

fare_cuts = [-1,12,50,100,1000]

fare_labels = ["0-12","12-50","50-100","100+"]

train["Fare_categories"] = pd.cut(train["Fare"],fare_cuts,labels=fare_labels)

holdout["Fare_categories"] = pd.cut(holdout["Fare"],fare_cuts,labels=fare_labels)





#categorize cabin types



train["Cabin_type"] = train["Cabin"].str[0]

train["Cabin_type"] = train["Cabin_type"].fillna("Unknown")

train = train.drop('Cabin',axis=1)



holdout["Cabin_type"] = holdout["Cabin"].str[0]

holdout["Cabin_type"] = holdout["Cabin_type"].fillna("Unknown")

holdout = holdout.drop('Cabin',axis=1)



#engineer Title feature

titles = {

        "Mr" :         "Mr",

        "Mme":         "Mrs",

        "Ms":          "Mrs",

        "Mrs" :        "Mrs",

        "Master" :     "Master",

        "Mlle":        "Miss",

        "Miss" :       "Miss",

        "Capt":        "Officer",

        "Col":         "Officer",

        "Major":       "Officer",

        "Dr":          "Officer",

        "Rev":         "Officer",

        "Jonkheer":    "Royalty",

        "Don":         "Royalty",

        "Sir" :        "Royalty",

        "Countess":    "Royalty",

        "Dona":        "Royalty",

        "Lady" :       "Royalty"

    }

train_titles = train["Name"].str.extract(' ([A-Za-z]+)\.',expand=False)

train["Title"] = train_titles.map(titles)



holdout_titles = holdout["Name"].str.extract(' ([A-Za-z]+)\.',expand=False)

holdout["Title"] = holdout_titles.map(titles)



#engineer isalone

familySize_train = train[["SibSp","Parch"]].sum(axis=1)

train["isalone"] = np.where(familySize_train>=1, 1, 0)



familySize_holdout = holdout[["SibSp","Parch"]].sum(axis=1)

holdout["isalone"] = np.where(familySize_holdout>=1, 1, 0)





#dummy variables for all the categorical features

def get_dummies(df, column_name):

    dummies = pd.get_dummies(df[column_name],prefix=column_name)

    df = pd.concat([df,dummies],axis=1)

    return df



columnNames = ["Age_categories", "Pclass", "Sex", "Fare_categories", "Title", "Cabin_type", "Embarked"]



for column in columnNames:

    dummies_train = pd.get_dummies(train[column],prefix=column)

    train = pd.concat([train,dummies_train],axis=1)

    

    dummies_holdout = pd.get_dummies(holdout[column],prefix=column)

    holdout = pd.concat([holdout,dummies_holdout],axis=1)

    

train.head(5)

holdout.head(5)



print(holdout.columns)



columns = ['Age_categories_Missing', 'Age_categories_Infant',

       'Age_categories_Child', 'Age_categories_Teenager',

       'Age_categories_Young Adult', 'Age_categories_Adult',

       'Age_categories_Senior', 'Pclass_1', 'Pclass_2', 'Pclass_3',

       'Sex_female', 'Sex_male', 'Embarked_C', 'Embarked_Q', 'Embarked_S',

       'Fare_categories_0-12',

       'Fare_categories_12-50','Fare_categories_50-100', 'Fare_categories_100+',

       'Title_Master', 'Title_Miss', 'Title_Mr','Title_Mrs', 'Title_Officer',

       'Title_Royalty', 'Cabin_type_A','Cabin_type_B', 'Cabin_type_C', 'Cabin_type_D',

       'Cabin_type_E','Cabin_type_F', 'Cabin_type_G', 'Cabin_type_Unknown', 'isalone']



#model-selection

def get_model(df, features):

    

    train_X = df[features]

    train_y = df["Survived"]

    

    cv = ShuffleSplit(n_splits = 10, test_size = .3, train_size = .6, random_state = 0)

    model_params = [

            {

                    "name": "RandomForestClassifier",

                    "estimator": RandomForestClassifier(random_state=0),

                    "hyperparameters":

                        {

                                "n_estimators": [20,25, 35, 40,45, 50, 55, 60, 65, 70, 75],

                                "criterion": ["entropy", "gini"],

                                "max_features": ["log2", "sqrt"],

                                "min_samples_leaf": [1, 5, 8],

                                "min_samples_split": [2, 3, 5]

                        }

            },

            {

                    "name": "DecisionTreeClassifier",

                    "estimator": tree.DecisionTreeClassifier(),

                    "hyperparameters":

                        {

                                "criterion": ["entropy", "gini"],

                                "max_depth": [None, 2,4,6,8,10, 12, 14, 16],

                                'min_samples_split': [2,3,4,5,10,.03,.05,.1],

                                "max_features": [None, "auto"],

                                "min_samples_leaf": [1,2,3,4,5,10,12, .5, .03,.05,.1]

                        }

            },

            {

                    "name": "KernelSVMClassifier",

                    "estimator": SVC(random_state=0),

                    "hyperparameters":

                        {

                                "kernel": ["rbf"],

                                "C": np.logspace(-9, 3, 13),

                                "gamma": np.logspace(-9, 3, 13)

                        }

            } ,

            {

                    "name": "KNeighborsClassifier",

                    "estimator": KNeighborsClassifier(),

                    "hyperparameters":

                        {

                                "n_neighbors": range(1,20,2),

                                "weights": ["distance", "uniform"],

                                "algorithm": ["ball_tree", "kd_tree", "brute"],

                                "p": [1,2]

                        }

            },

            {

                    "name": "LogisticRegressionClassifier",

                    "estimator": LogisticRegression(),

                    "hyperparameters":

                        {

                                "solver": ["newton-cg", "lbfgs", "liblinear"]

                        }

            }          

            ]

    models = []

    for model in model_params:

        print(model["name"])

        grid = GridSearchCV(model["estimator"], 

                            param_grid=model["hyperparameters"], 

                            cv=10)

        grid.fit(train_X, train_y)

        

        model_att = {

                "model": grid.best_estimator_, 

                "best_params": grid.best_params_, 

                "best_score": grid.best_score_,

                "grid": grid

                }

        models.append(model_att)

        print("Evaluated model and its params: ")

        print(grid.best_params_)

        print(grid.best_score_)

    return models



#Artificial Neural Network

def ann_model(df, features):

    classifier = Sequential()

    classifier.add(Dense(input_dim=len(features), units=15, activation="relu", kernel_initializer="uniform"))

    

    classifier.add(Dense(units = 15, kernel_initializer = 'uniform', activation = 'relu'))

    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

    

    train_X = df[features]

    train_y = df["Survived"]

    classifier.fit(np.array(train_X), np.array(train_y), batch_size = 10, epochs = 100)

    return classifier



#feature selection using RFECV

def get_features(df, columns, model=None):

    newDf = df.copy()

    newDf = newDf.select_dtypes(['number'])

    newDf = newDf.dropna(axis=1, how='any')

    

    #dropColumns = ["PassengerId", "Survived"]

    #newDf = newDf.drop(dropColumns, axis = 1)

    

    all_X = newDf[columns]

    all_y = df["Survived"]

    

    cv = StratifiedShuffleSplit(n_splits = 10, test_size = .3, train_size = .6, random_state = 0)

    if model == None:

        classifier = tree.DecisionTreeClassifier(

                criterion="entropy",

                max_depth=10,

                max_features='auto',

                max_leaf_nodes=None,

                min_impurity_decrease = 0.0,

                min_samples_leaf = 10,

                min_samples_split = 3

                )

    else:

        classifier = model

    selector = RFECV(classifier, scoring = 'roc_auc', cv=cv, step = 1)

    selector.fit(all_X,all_y)

    rfecv_columns = all_X.columns[selector.support_]

    return rfecv_columns



models = get_model(train, columns)



#select the best one based on its index from console

best_grid = models[0]["grid"]

best_classifier = models[0]["model"]

best_params = models[0]["best_params"]



rfecv_features = get_features(train, columns, best_classifier)

print(len(rfecv_features))

print(rfecv_features)



models = get_model(train, rfecv_features)

best_classifier = models[0]["model"]



predictions = best_classifier.predict(holdout[rfecv_features])



sub = {"PassengerId": holdout["PassengerId"], "Survived": predictions}

submission = pd.DataFrame(sub)

submission.to_csv(path_or_buf="Submission.csv", index=False, header=True)


