# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

# data analysis and wrangling

import pandas as pd

import numpy as np

import random as rnd



# visualization

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.neural_network import MLPClassifier

from sklearn.svm import SVC

from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, learning_curve



# machine learning

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import Perceptron

from sklearn.linear_model import SGDClassifier

from sklearn.tree import DecisionTreeClassifier

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import StratifiedKFold

train = pd.read_csv("../input/titanic/train.csv")

test = pd.read_csv("../input/titanic/test.csv")

train = train.drop(['Name','Ticket','Embarked','Cabin'], axis=1)

test    = test.drop(['Name','Ticket','Embarked','Cabin'], axis=1)

# Fare

# get average, std, and number of NaN values in titanic_df

average_age_titanic   = train["Age"].mean()

std_age_titanic       = train["Age"].std()

count_nan_age_titanic = train["Age"].isnull().sum()



# get average, std, and number of NaN values in test_df

average_age_test   = test["Age"].mean()

std_age_test       = test["Age"].std()

count_nan_age_test = test["Age"].isnull().sum()





# generate random numbers between (mean - std) & (mean + std)

rand_1 = np.random.randint(average_age_titanic - std_age_titanic, average_age_titanic + std_age_titanic, size = count_nan_age_titanic)

rand_2 = np.random.randint(average_age_test - std_age_test, average_age_test + std_age_test, size = count_nan_age_test)

# only for test_df, since there is a missing "Fare" values

test["Fare"].fillna(test["Fare"].median(), inplace=True)



# convert from float to int

train['Fare'] = train['Fare'].astype(int)

test['Fare']    = train['Fare'].astype(int)

# NOTE: drop all null values, and convert to int





# fill NaN values in Age column with random values generated

train["Age"][np.isnan(train["Age"])] = rand_1

test["Age"][np.isnan(test["Age"])] = rand_2



# convert from float to int

test['Age'] = train['Age'].astype(int)

test['Age']    = test['Age'].astype(int)

train["Sex"] = train["Sex"].map({"male": 0, "female":1})

test["Sex"] = test["Sex"].map({"male": 0, "female":1})
X_train = train.drop("Survived",axis=1)

Y_train = train["Survived"]

X_test  = test.drop("PassengerId",axis=1).copy()
# Logistic Regression



#logreg = LogisticRegression()



#logreg.fit(X_train, Y_train)



#Y_pred = logreg.predict(X_test)



#logreg.score(X_train, Y_train)
# Random Forests



#random_forest = RandomForestClassifier(n_estimators=100)



#random_forest.fit(X_train, Y_train)



#Y_pred = random_forest.predict(X_test)



#random_forest.score(X_train, Y_train)
# Cross validate model with Kfold stratified cross val

kfold = StratifiedKFold(n_splits=10)
# Modeling step Test differents algorithms 

random_state = 2

classifiers = []

classifiers.append(SVC(random_state=random_state))

classifiers.append(DecisionTreeClassifier(random_state=random_state))

classifiers.append(AdaBoostClassifier(DecisionTreeClassifier(random_state=random_state),random_state=random_state,learning_rate=0.1))

classifiers.append(RandomForestClassifier(random_state=random_state))

classifiers.append(ExtraTreesClassifier(random_state=random_state))

classifiers.append(GradientBoostingClassifier(random_state=random_state))

classifiers.append(MLPClassifier(random_state=random_state))

classifiers.append(KNeighborsClassifier())

classifiers.append(LogisticRegression(random_state = random_state))

classifiers.append(LinearDiscriminantAnalysis())

cv_results = []

for classifier in classifiers :

    cv_results.append(cross_val_score(classifier, X_train, y = Y_train, scoring = "accuracy", cv = kfold, n_jobs=4))



cv_means = []

cv_std = []

for cv_result in cv_results:

    cv_means.append(cv_result.mean())

    cv_std.append(cv_result.std())



cv_res = pd.DataFrame({"CrossValMeans":cv_means,"CrossValerrors": cv_std,"Algorithm":["SVC","DecisionTree","AdaBoost",

"RandomForest","ExtraTrees","GradientBoosting","MultipleLayerPerceptron","KNeighboors","LogisticRegression","LinearDiscriminantAnalysis"]})



g = sns.barplot("CrossValMeans","Algorithm",data = cv_res, palette="Set3",orient = "h",**{'xerr':cv_std})

g.set_xlabel("Mean Accuracy")

g = g.set_title("Cross validation scores")
### META MODELING  WITH ADABOOST, RF, EXTRATREES and GRADIENTBOOSTING



# Adaboost

#DTC = DecisionTreeClassifier()



#adaDTC = AdaBoostClassifier(DTC, random_state=7)



#ada_param_grid = {"base_estimator__criterion" : ["gini", "entropy"],

#              "base_estimator__splitter" :   ["best", "random"],

#              "algorithm" : ["SAMME","SAMME.R"],

#              "n_estimators" :[1,2],

#              "learning_rate":  [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3,1.5]}



#gsadaDTC = GridSearchCV(adaDTC,param_grid = ada_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)



#gsadaDTC.fit(X_train,Y_train)



#ada_best = gsadaDTC.best_estimator_

#ExtraTrees 

#ExtC = ExtraTreesClassifier()





## Search grid for optimal parameters

#ex_param_grid = {"max_depth": [None],

#              "max_features": [1, 3, 10],

#              "min_samples_split": [2, 3, 10],

#              "min_samples_leaf": [1, 3, 10],

#              "bootstrap": [False],

#              "n_estimators" :[100,300],

#              "criterion": ["gini"]}





#gsExtC = GridSearchCV(ExtC,param_grid = ex_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)



#gsExtC.fit(X_train,Y_train)



#ExtC_best = gsExtC.best_estimator_



# Best score

#gsExtC.best_score_

# RFC Parameters tunning 

#RFC = RandomForestClassifier()





## Search grid for optimal parameters

#rf_param_grid = {"max_depth": [None],

#              "max_features": [1, 3, 10],

#              "min_samples_split": [2, 3, 10],

#              "min_samples_leaf": [1, 3, 10],

#              "bootstrap": [False],

#              "n_estimators" :[100,300],

#              "criterion": ["gini"]}





#gsRFC = GridSearchCV(RFC,param_grid = rf_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)



#gsRFC.fit(X_train,Y_train)



#RFC_best = gsRFC.best_estimator_



# Best score

#gsRFC.best_score_
#def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,

#                        n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 5)):

#    """Generate a simple plot of the test and training learning curve"""

#    plt.figure()

#    plt.title(title)

#    if ylim is not None:

#        plt.ylim(*ylim)

#    plt.xlabel("Training examples")

#    plt.ylabel("Score")

#    train_sizes, train_scores, test_scores = learning_curve(

#        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)

#    train_scores_mean = np.mean(train_scores, axis=1)

#    train_scores_std = np.std(train_scores, axis=1)

#    test_scores_mean = np.mean(test_scores, axis=1)

#    test_scores_std = np.std(test_scores, axis=1)

#    plt.grid()



#    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,

#                     train_scores_mean + train_scores_std, alpha=0.1,

#                     color="r")

#    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,

#                     test_scores_mean + test_scores_std, alpha=0.1, color="g")

#    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",

#             label="Training score")

#    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",

#             label="Cross-validation score")



#    plt.legend(loc="best")

#    return plt

#g = plot_learning_curve(gsRFC.best_estimator_,"RF mearning curves",X_train,Y_train,cv=kfold)

#g = plot_learning_curve(gsExtC.best_estimator_,"ExtraTrees learning curves",X_train,Y_train,cv=kfold)

#g = plot_learning_curve(gsSVMC.best_estimator_,"SVC learning curves",X_train,Y_train,cv=kfold)

#g = plot_learning_curve(gsadaDTC.best_estimator_,"AdaBoost learning curves",X_train,Y_train,cv=kfold)

#g = plot_learning_curve(gsGBC.best_estimator_,"GradientBoosting learning curves",X_train,Y_train,cv=kfold)




#GBC = GradientBoostingClassifier()

#gb_param_grid = {'loss' : ["deviance"],

#              'n_estimators' : [100,200,300],

#              'learning_rate': [0.1, 0.05, 0.01],

#              'max_depth': [4, 8],

#              'min_samples_leaf': [100,150],

#              'max_features': [0.3, 0.1] 

#              }



#gsGBC = GridSearchCV(GBC,param_grid = gb_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)



#gsGBC.fit(X_train,Y_train)



#GBC_best = gsGBC.best_estimator_



# Best score

#gsGBC.best_score_

### SVC classifier

#SVMC = SVC(probability=True)

#svc_param_grid = {'kernel': ['rbf'], 

#                  'gamma': [ 0.001, 0.01, 0.1, 1],

#                  'C': [1, 10, 50, 100,200,300, 1000]}



#gsSVMC = GridSearchCV(SVMC,param_grid = svc_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)



#gsSVMC.fit(X_train,Y_train)



#SVMC_best = gsSVMC.best_estimator_



# Best score

#gsSVMC.best_score_
DTC = DecisionTreeClassifier()

ExtC = ExtraTreesClassifier()

RFC = RandomForestClassifier()

LDR=LogisticRegression()

LR=LinearDiscriminantAnalysis()

GBC=GradientBoostingClassifier()
votingC = VotingClassifier(estimators=[('rfc', RFC), ('extc', ExtC),

('dtc', DTC), ('logr',LDR),('gbc',GBC),('lr',LR)], voting='soft', n_jobs=4)



votingC = votingC.fit(X_train, Y_train)
train.shape

test_Survived = pd.Series(votingC.predict(test), name="Survived")



#results = pd.concat([IDtest,test_Survived],axis=1)



#results.to_csv("ensemble_python_voting.csv",index=False)

output = pd.DataFrame({'PassengerId': test['PassengerId'],

                     'Survived': test_Survived})

output.to_csv('submission.csv', index=False)