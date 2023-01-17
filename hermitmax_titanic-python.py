# data analysis and wrangle

import pandas as pd 

import numpy as np

from collections import Counter

from sklearn import preprocessing



# visualisation 

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



# machine learning

from sklearn.cluster import KMeans

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import Perceptron

from sklearn.linear_model import SGDClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.pipeline import Pipeline

from sklearn.model_selection import GridSearchCV
# step1: load data 

train_data = pd.read_csv("../input/train.csv")

test_data = pd.read_csv("../input/test.csv")

combine = [train_data,test_data]
# divide data into categorical and numeric variables

# -> so we know how to visualize them later

train_data.head()

#print (pd.unique(train_data.loc[:,"Embarked"]))

#print (pd.unique(test_data.loc[:,"Embarked"]))

# categorical variable

#     nominal var: PassengeId[int], Name[char], Sex[male/female], Ticket[char+int], cabin[NaN]

#     ordinal var: Pclass[int], Embarked[char]

# numerical variable

#     discrete var: Age, SibSp, Parch

#     continuous var: Fare

# Label

#     survived: 0 -- not survived, 1 -- survived
# any null, outlier or abnormal feature or pattern or representative

# -> the reason why we need to find them is because

# -> ML models do not like missing values and will be confused with outlier

#    and also if features are representative, it might contribute a lot for our model

print (train_data.info())

print ("-"*40)

print (test_data.info())

# train - cabin(a lot of null values -> think if drop them.)

#       - Embarked (2 null values -> replace with similar values)

#       - Age (200+ null values --> replace with similar values)

# test - cabin (a lot of null values -> think if drop them.)

#      - Fare (1 null values -> replace with median or mean)
# check representativeness of features

# my purpose: if the feature is representative (-> feature is useful)

#             e.g. like Pclass, if Pclass = 1 has high survival (> average), and others have lower rates, 

#             then this feature may have correlations with survival rates --> so keep it. 

train_data.head()

train_data.describe(percentiles=[.1, .2, .3, .4, .5, .6, .7, .8, .9, 1.])

## PassengerId --> unique ID from 1 to 891 --> not representative --> drop them

# Survived --> 38% survival rate 

# Pclass --> most of people (50%+) from Pclass 3 --> survial rate for different Pclass

# Age --> has missing values & 80% people below 41 years old --> survival rate for different ages

# SibSp --> most of people (60%+) does not have brothers & sisters together -> look correlations

# Parch --> more than 70% does have parents together -> look correlations

# Fare --> most of people (80%) pay below 39 but there are weired values like 512(?) & 0(staff).

#      --> see correlations first & may build new ratio variables based on it

#      --> too many unique Fares --> need to build ratio variables
train_data.describe(include =["O"])

# Name -> unique -> it is useless now --> may group similar name together

# Sex -> most of people (577/891) are male --> see correlations

# Ticket -> some ticket number are same --> why?

## Cabin -> some Cabin are same -> some people share one room

#        -> not only has too many missing value and also it might have no correlations with labels

# Embarked -> people from three ports & 644/889 from S
# Assumption

# group one: high level customers --> [Pclass, Fare] --> high survival rate

# group two: have relatives on boat --> [SibSp, Parch] --> high survival rate

# group three: lady first --> Sex --> lady has high survival rate

# group four: child, elder first --> Age --> have high survival rate

# group five: from different ports --> Embarked --> surival rate varying
# group one - true

train_data[["Pclass", "Survived"]].groupby("Pclass", as_index = False).mean().sort_values(by = "Survived", ascending = False)

#train_data[["Fare", "Survived"]].groupby("Fare", as_index = False).mean().sort_values(by = "Survived", ascending = False)

# try later about Fare - maybe make a range variable
#g = sns.FacetGrid(train_data)

#g.map(sns.pointplot, "Pclass", "Survived", palette='deep')
# group two - true

print (train_data[["SibSp", "Survived"]].groupby("SibSp", as_index = False).mean().sort_values(by = "Survived", ascending = False))

print ("-"*40)

print (train_data[["Parch", "Survived"]].groupby("Parch", as_index = False).mean().sort_values(by = "Survived", ascending = False))

# it seems like when you have too high or no SibSp (>=3 or =0) -> below average survival rate

# it seems like when you have too high or no Parch (>=4 or =0) -> below average survival rate

# if you have too many family memeber, you dont have enough power to help every one and they too

# and also, if you do have family memeber, nobody help you

# also, maybe we can combine those two variables later
# group three - true

train_data[["Sex", "Survived"]].groupby("Sex", as_index = False).mean().sort_values(by = "Survived", ascending = False)
# group four - later -> need to re-design variable

g = sns.FacetGrid(train_data, col = "Survived")

g.map(plt.hist, "Age")

# not obvious -> create Age range

train_data["AgeRange"] = pd.cut(train_data["Age"], 5)

train_data[["AgeRange", "Survived"]].groupby("AgeRange", as_index = False).mean().sort_values(by = "Survived", ascending = False)
# group five - true assumption

train_data[["Embarked", "Survived"]].groupby("Embarked", as_index = False).mean().sort_values(by = "Survived", ascending = False)

# people from C port have high survial rate
train_data = pd.read_csv("../input/train.csv")

train_label = train_data["Survived"]

train_data = train_data.drop("Survived", axis = 1)

test_data = pd.read_csv("../input/test.csv")

combine = [train_data,test_data]

train_data.head()
# complete incomplete features

# train 

#       - Embarked (2 null values -> replace with similar values)

#       - Age (200+ null values --> replace with similar values)

# test 

#      - Fare (1 null values -> replace with median or mean)



# correct - if outlier drop them / remove abnormal data

#         - PassengerId[int] ————— drop 

#         - Name[char] ————— drop

#         - cabin[NaN] ————— drop



# convert categorical variables into numeric variables/dummy variables/one-hot

#         - Sex[male/female] ————— dummy 

#         - Ticket[char+int] ————— dummy 

#         - Pclass[int] ————— dummy 

#         - Embarked[char] ————— dummy 



# create  - build new variables based on existing varialbes
# correct features

print ("Before", train_data.shape, test_data.shape, combine[0].shape, combine[1].shape)



train_df = train_data.drop(["PassengerId", "Name", "Cabin", "Ticket"], axis = 1)

test_df = test_data.drop(["PassengerId", "Name", "Cabin", "Ticket"], axis = 1)

del train_data 

del test_data

combine = [train_df, test_df]



print ("After", train_df.shape, test_df.shape, combine[0].shape, combine[1].shape)
# convert features

categories_to_dummies = ["Sex", "Pclass", "Embarked"]

for i, df in enumerate(combine):

    for j in categories_to_dummies:

        # train

        # categories to dummies 

        tmp = pd.get_dummies(df[j], prefix=j)

        df = df.join(tmp)

        df = df.drop(j, axis = 1)

        combine[i] = df

del tmp

del df

print ("After", combine[0].shape, combine[1].shape)

combine[0].head()
# handle missing value

# Fare - test 

# better solution

# test_df['Fare'].fillna(test_df['Fare'].dropna().median(), inplace=True)

tmp_test = combine[1]

tmp_test.loc[tmp_test["Fare"].isnull(), "Fare"] = tmp_test["Fare"].median()

pd.isnull(tmp_test["Fare"]).sum() > 0
train_df["AgeRange"] = pd.cut(train_df["Age"], 5)
for i, df in enumerate(combine):

    df.loc[df["Age"] <= 16.336, "Age"] = 0

    df.loc[(df["Age"] > 16.336) & (df["Age"] <=32.252), "Age"] = 1

    df.loc[(df["Age"] > 32.252) & (df["Age"] <=48.168), "Age"] = 2

    df.loc[(df["Age"] > 48.168) & (df["Age"] <=64.084), "Age"] = 3

    df.loc[(df["Age"] > 64.084) & (df["Age"] <=80.0), "Age"] = 4

    combine[i] = df

del df

print (combine[0]["Age"])

print (combine[1]["Age"])
# clustering

# build temporary train & test sets

tmp_train = combine[0].copy()

tmp_test = combine[1].copy()

tmp_train = tmp_train.drop("Age", axis = 1)

tmp_test = tmp_test.drop("Age", axis = 1)

# kmeans clustering for train & test sets

# & assign "kmeans_labels" back to "combine sets"

kmeans = KMeans(n_clusters=5, random_state=0).fit(tmp_train)

combine[0]["kmeans_labels"] = pd.Series(kmeans.labels_)

combine[1]["kmeans_labels"] = pd.Series(kmeans.predict(tmp_test))

# verify success or not

combine[1].head()
# replace age's missing value 

tmp_train = combine[0].copy()

tmp_test = combine[1].copy()

tmp_age_train = []

tmp_age_test = []

for i in range(5):

    tmp_age_train.append(Counter(tmp_train.loc[tmp_train["kmeans_labels"] == i, "Age"]).most_common()[0][0])

    tmp_age_test.append(Counter(tmp_test.loc[tmp_test["kmeans_labels"] == i, "Age"]).most_common()[0][0])

for j in range(5):

    tmp_train.loc[(tmp_train["kmeans_labels"] == j) & (tmp_train["Age"].isnull()), "Age"] = tmp_age_train[j]

    tmp_test.loc[(tmp_test["kmeans_labels"] == j) & (tmp_test["Age"].isnull()), "Age"] = tmp_age_test[j]

combine[0]["Age"] = tmp_train["Age"]

combine[1]["Age"] = tmp_test["Age"]

print (pd.isnull(combine[0]["Age"]).sum() >0, pd.isnull(combine[1]["Age"]).sum() >0)
# modeling

# drop, kmeans, since it is not original features

X_train = combine[0].drop("kmeans_labels", axis = 1)

Y_train = train_label

X_test = combine[1].drop("kmeans_labels", axis = 1)

print ("After", X_train.shape, Y_train.shape, X_test.shape)

X_train.head()
# split original train dataset into train & dev sets

from sklearn.model_selection import train_test_split

X_train_t, X_train_dev, y_train_t, y_train_dev = train_test_split(

     X_train, Y_train, test_size=0.3, random_state=2)

print ("train_train", X_train_t.shape, y_train_t.shape)

print ("train_dev", X_train_dev.shape, y_train_dev.shape)

#X_train.iloc[8, :]

#Y_train[8]

#y_train_t[9]

#X_train.iloc[9, :]

#Y_train.iloc[9]
# group 1: perceptual, logistic regression, SVM, neural network, kNN
X_train_t.head()
from sklearn import preprocessing

# build 

min_max_scaler = preprocessing.MinMaxScaler()

# scale train data

copy_that_t = X_train_t.copy()

scaled_X_t = min_max_scaler.fit_transform(X_train_t[["Age", "SibSp", "Parch", "Fare"]])

copy_that_t[["Age", "SibSp", "Parch", "Fare"]] = pd.DataFrame(scaled_X_t, columns=["Age", "SibSp", "Parch", "Fare"], index = copy_that_t.index)

X_train_t = copy_that_t

# scale train dev data

copy_that_dev = X_train_dev.copy()

scaled_X_dev = min_max_scaler.transform(copy_that_dev[["Age", "SibSp", "Parch", "Fare"]])

copy_that_dev[["Age", "SibSp", "Parch", "Fare"]] = pd.DataFrame(copy_that_dev, columns=["Age", "SibSp", "Parch", "Fare"], index = copy_that_dev.index)

X_train_dev = copy_that_dev

#scaled_X_dev = min_max_scaler.transform(X_train_dev[["Age", "SibSp", "Parch", "Fare"]])

#X_train_dev[["Age", "SibSp", "Parch", "Fare"]] = pd.DataFrame(scaled_X_dev, columns=["Age", "SibSp", "Parch", "Fare"], index = X_train_dev.index)

X_train_t.head()

X_train_dev.head()
# logistic regression

# tune parameters

# keep cv, dual, penalty, solver, refit, random_state same

# --> then tune Cs to reach the highest train_cv_acc with very large max_iter

# --> then tune max_iter to reach the highest train_cv_acc with optimal Cs

from sklearn.linear_model import LogisticRegressionCV





# build pipe

# use pipe, you can only use two operations: fit and transform. 

# --> it is not suitable for my sitation, i need to use fit_transforms for first four columns

# --> then to do the fit for all the data

logCV = LogisticRegressionCV(Cs=100, cv=5, dual=False, penalty='l2', solver='lbfgs', max_iter=50,

                            refit=True, random_state=1)

# set parameters

parameters = dict(

Cs = [100, 120, 130], 

cv = [5], 

max_iter = [10, 15]

)



# estimators

estimator = GridSearchCV(logCV,parameters)

estimator.fit(X_train_t, y_train_t)

#logistic.predict(X_test)

print ("best estimator", estimator.best_estimator_, estimator.best_params_)

#print ("best coef", pipe.named_steps["logCVCV"].coef_)

#print ("train_cv_acc", pipe.named_steps["logCVCV"].scores_)

print ("train_cv_acc", estimator.score(X_train_t, y_train_t)*100)

print ("dev_acc", estimator.score(X_train_dev, y_train_dev)*100)

# TO DO

# best estimator from GridSearchCV

# --> LogisticRegressionCV(best_parameters)

# --> logCV.fit(X_train_t, y_trian_t)

#names = ['Age', 'SibSp', 'Parch', 'Fare', 'Sex_female', 'Sex_male', 'Pclass_1',

#       'Pclass_2', 'Pclass_3', 'Embarked_C', 'Embarked_Q', 'Embarked_S']

#plot_coef_df = pd.DataFrame({"names":np.array(names), 

#                             "logCV_coef":np.array(logCV.coef_.reshape(12,1))

#                            })

# pipe

#pipe = Pipeline(steps=[("logCVCV", logCV)])

#pipe.set_params(logCVCV__Cs = [4, 5, 6, 7, 8, 9,10], logCVCV__cv = 5, 

                #logCVCV__max_iter = 30, logCVCV__random_state=1).fit(X_train_t, y_train_t)

#logCV.fit(X_train_t, y_train_t)
# SVM

from sklearn.svm import SVC

svm = SVC(C=1.0, kernel='rbf', degree=5, gamma='auto', coef0=0.0, 

           shrinking=True, probability=False, tol=0.001, 

           cache_size=200, class_weight=None, verbose=False, max_iter=-1, 

           decision_function_shape='ovr', random_state=2)



# set parameters

parameters_svm = dict(

C = [1.0, 1.5, 2.0, 2.5]

)



# build estimators

estimator_svm = GridSearchCV(svm,parameters_svm, cv = 5)

estimator_svm.fit(X_train_t, y_train_t)

print ("best estimator_svm", estimator_svm.best_estimator_, estimator_svm.best_params_)

print ("train_cv_acc", estimator_svm.score(X_train_t, y_train_t)*100)

print ("dev_acc", estimator_svm.score(X_train_dev, y_train_dev)*100)
# NN

from sklearn.neural_network import MLPClassifier

NN = MLPClassifier(hidden_layer_sizes=(100, ), activation='relu', solver='lbfgs', 

                   alpha=0.1, batch_size='auto', learning_rate='constant', 

                   learning_rate_init=0.001, power_t=0.5, 

                   max_iter=500, shuffle=True, random_state=3, tol=0.0001, 

                   verbose=False, warm_start=False, momentum=0.9, 

                   nesterovs_momentum=True, early_stopping=True, validation_fraction=0.1, 

                   beta_1=0.9, beta_2=0.999, epsilon=1e-08)

# set parameters

parameters_NN = {

    'hidden_layer_sizes':[(10, )],

    'activation':['relu'],

    'alpha':[0.5, 0.4, 1.0, 1.2, 1.5],

    'solver':['lbfgs']

}

# build estimators

estimator_NN = GridSearchCV(NN,parameters_NN, cv = 5)

estimator_NN.fit(X_train_t, y_train_t)

print ("best estimator_svm", estimator_NN.best_estimator_, estimator_NN.best_params_)

print ("train_cv_acc", estimator_NN.score(X_train_t, y_train_t)*100)

print ("dev_acc", estimator_NN.score(X_train_dev, y_train_dev)*100)
from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=10, shuffle=False, random_state=1)

KFoldSplit = skf.split(X_train_t, y_train_t)
print (X_train_t.shape, y_train_t.shape)

skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)

for i, j in skf.split(X_train_t, y_train_t):

    #print (i)

    #print (j)

    #print (np.unique(y_train_t.iloc[i]))

    break

#X_train_dev.iloc[test_index,:]
# kNN

from sklearn.neighbors import KNeighborsClassifier



scores = []

tmpScore = 0

for j in range(1,20, 1):

    for train_index, test_index in skf.split(X_train_t, y_train_t):

        kNN = KNeighborsClassifier(n_neighbors=j, weights='uniform', 

                           algorithm='auto', leaf_size=30, p=1, 

                           metric='minkowski', metric_params=None, 

                           n_jobs=1)

        X = X_train_t[train_index,:]

        y = y_train_t[train_index]

        kNN.fit(X, y)

        tmpScore += (kNN.score(X_train_dev[test_index,:], y_train_dev[test_index])/10.0)

    scores.append(tmpScore)

    tmpScore = 0

x = range(1, 20, 1)

plt.plot(x, scores)

plt.show()

#parameters_kNN = {

#    'n_neighbors':[3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16], 

#    'weights': ['distance', 'uniform'], 

#    'algorithm':['auto'], 

#    'leaf_size':[30],

#    'p': [1, 2, 3],

#}

# build estimators

#estimator_kNN = GridSearchCV(kNN,parameters_kNN, cv = 5)

#estimator_kNN.fit(X_train_t, y_train_t)

#print ("best estimator_svm", estimator_kNN.best_estimator_, estimator_kNN.best_params_)

#print ("train_cv_acc", estimator_kNN.score(X_train_t, y_train_t)*100)

#print ("dev_acc", estimator_kNN.score(X_train_dev, y_train_dev)*100)

#????
# perceptual 

from sklearn.linear_model import Perceptron

per = Perceptron(penalty='l1', alpha=0.0001, fit_intercept=True, 

                 max_iter=500, tol=0.00001, shuffle=True, verbose=0, 

                 eta0=1.0, n_jobs=1, random_state=5, class_weight=None, 

                 warm_start=False, n_iter=None

)

# set parameters

parameters_per = {

    'penalty':['l2', 'l1'], 

    'alpha':[0.0005, 0.001, 0.01, 0.5],

}

# build estimators

estimator_per = GridSearchCV(per,parameters_per, cv = 10)

estimator_per.fit(X_train_t, y_train_t)

print ("best estimator_svm", estimator_per.best_estimator_, estimator_per.best_params_)

print ("train_cv_acc", estimator_per.score(X_train_t, y_train_t)*100)

print ("dev_acc", estimator_per.score(X_train_dev, y_train_dev)*100)
# randome forest 

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=10, criterion='gini', 

                            max_depth=None, min_samples_split=2, 

                            min_samples_leaf=1, min_weight_fraction_leaf=0.0, 

                            max_features='auto', max_leaf_nodes=None, 

                            min_impurity_decrease=0.0, min_impurity_split=None, 

                            bootstrap=True, oob_score=False, n_jobs=1, 

                            random_state=6, verbose=0, warm_start=False, 

                            class_weight=None)

# set parameters

params_rf = {

    'n_estimators':[10,20, 25,30],

    'max_depth':[None, 3, 4],

    'min_samples_split':[2],

    'min_samples_leaf':[1],

    'max_features':[5, 10],

    'max_leaf_nodes':[None]

}

# build estimators

estimator_rf = GridSearchCV(rf,params_rf, cv = 5)

estimator_rf.fit(X_train_t, y_train_t)

print ("best estimator_rf", estimator_rf.best_estimator_, estimator_rf.best_params_)

print ("train_cv_acc", estimator_rf.score(X_train_t, y_train_t)*100)

print ("dev_acc", estimator_rf.score(X_train_dev, y_train_dev)*100)